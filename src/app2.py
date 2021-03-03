import multiprocessing as mp
import random
from enum import Enum
from queue import Empty
from time import sleep
import math

from src.system.periscope import *

from src.parser import parse
from src.render import Renderer, pygame
from src.system.periscope import Periscope, MirrorLocation, Target


class SolveAlgorithm(Enum):
    DIRECT = 1
    NEURAL_NET = 2


class TargetMoveMode(Enum):
    KEYBOARD = 1
    RANDOM_MOVE = 2


class PeriscopeApplication:

    def __init__(
            self,
            input_model: str = '2d',
            algorithm: SolveAlgorithm = SolveAlgorithm.DIRECT,
            window_size=3,
            lose_prob=0.2,
            transfer_number=13
    ):
        self.log_list = []
        self.count_win = 0
        self.count_all = 0
        self.sleep_time = 1
        pygame.init()
        self.input_model = input_model
        config = parse(input_model)

        self.periscope: Periscope = Periscope(config)
        p_target = self.periscope.ray_to_aim().intersect_plane(
            Triangle(Point3d(0.2, 0.5, 0.2),
                     Point3d(0.2, 0.4, 0.1),
                     Point3d(0.2, 0.3, 0.5)
                     ))

        p_target_prev = self.periscope.ray_to_aim().intersect_plane(
            Triangle(Point3d(0.2, 0.5, 0.2),
                     Point3d(0.2, 0.4, 0.1),
                     Point3d(0.2, 0.3, 0.5)
                     ))
        tee = Target(p_target, config["target_radius"])
        tee_prev = Target(p_target_prev, config["prev_target_radius"])

        self.periscope.set_target(tee)
        self.periscope.set_prev_target(tee_prev)

        self.renderer = Renderer(self.periscope)

        self._sender_queue = mp.Queue()  # for target coord
        # self._receiver_queue = mp.Queue()  # for planes

        self.pack_number = 0  # number or package that caught for session (used in number mode)

        self.target_locations_x = mp.Array('d', transfer_number * window_size)
        self.target_locations_y = mp.Array('d', transfer_number * window_size)
        self.ii = 0
        self.target_locations_x[self.ii] = self.periscope.target.location.x
        self.target_locations_y[self.ii] = self.periscope.target.location.y
        self.ii += 1

        # Shared memory
        self.down_plane_points = mp.Array('d', 6)
        self.up_plane_points = mp.Array('d', 6)
        self.__init_share_memory()

        self.up_plane_queue = mp.Queue()
        self.down_plane_queue = mp.Queue()

        self._sender_process = mp.Process(target=self._gbn_sender_number,
                                          args=(self._sender_queue, self.up_plane_queue,
                                                window_size, lose_prob, transfer_number))

        self.up_plane_process: mp.Process = mp.Process(target=PeriscopeApplication.plane_direct_process,
                                                       args=(self.up_plane_queue, self.up_plane_points, self.periscope,
                                                             MirrorLocation.UP, self.target_locations_x,
                                                             self.target_locations_y, self._sender_queue, lose_prob, self.down_plane_queue))
        self.down_plane_process: mp.Process = mp.Process(target=PeriscopeApplication.plane_direct_process_2,
                                                         args=(
                                                             self.down_plane_queue, self.down_plane_points, self.periscope,
                                                             MirrorLocation.DOWN
                                                             ))

    class SenderArgs:
        def __init__(self, window_size, lose_prob, Sb=0):
            self.Sn = 0
            self.Sm = window_size
            self.window_size = window_size
            self.need_check = False
            self.lose_prob = lose_prob
            self.Sb = Sb

    def get_target_index_in_locations(self, target: Target):
        for i in range(self.target_locations_x.__sizeof__()):
            if target.location.x == self.target_locations_x[i] and \
                    target.location.y == self.target_locations_y[i]:
                return i

        return None

    @staticmethod
    def is_equals(t1: Target, t2: Target):
        return t1.get_description() == t2.get_description()

    @staticmethod
    def _send(queue_up: mp.Queue(), target: Target, lose_prob):
        r = random.random()

        if r >= lose_prob:
            queue_up.put(target)


    def _gbn_sender_number(self, sender_queue, receiver_up_queue, window_size, lose_prob,
                           pack_number):
        args = PeriscopeApplication.SenderArgs(window_size, lose_prob)
        while args.Sn <= pack_number:
            self._gbn_sender(sender_queue, receiver_up_queue, args)

    def _gbn_sender(self, self_queue: mp.Queue(), dist_up_queue: mp.Queue(),
                    args: SenderArgs):
        repeat = False
        sleep(self.sleep_time)
        print_str = 'Передатчик:'

        if args.Sn < args.Sm:
            target_message = Target(
                init_coords=Point3d(x=self.target_locations_x[args.Sn], y=self.target_locations_y[args.Sn], z=0),
                radius=0.02)
            PeriscopeApplication._send(queue_up=dist_up_queue, target=target_message, lose_prob=args.lose_prob)
            print_str += ' послан pkt: ' + target_message.get_description()
            args.Sn += 1

        else:
            try:
                target_message_number = self_queue.get()
                print_str += ' принят ACK ' + target_message_number.get_description() + \
                             f' Sb: {args.Sb} ' + 'x = ' + format(self.target_locations_x[args.Sb], '.2f') + ' y = ' + format(
                        self.target_locations_y[args.Sb], '.2f')

                target_Sb = Target(init_coords=Point3d(self.target_locations_x[args.Sb],
                                                                       self.target_locations_y[args.Sb]), radius=0.02)
                #print(f'Target Sb: {target_Sb}')
                #print(f'Target message: {target_message_number}')

                if PeriscopeApplication.is_equals(target_message_number, target_Sb):
                    target_Sn = Target(init_coords=Point3d(self.target_locations_x[args.Sn],
                                                                          self.target_locations_y[args.Sn], z=0),
                                                      radius=0.02)
                    PeriscopeApplication._send(dist_up_queue,
                                               target_Sn,
                                               args.lose_prob)
                    print_str += ' послан pkt: ' + target_Sn.get_description()
                    args.Sn, args.Sb, args.Sm = args.Sn + 1, args.Sb + 1, args.Sm + 1

                elif self.get_target_index_in_locations(target_message_number) > args.Sb:
                    repeat = True
            except Empty:
                repeat = True

        if repeat:
            args.Sn = args.Sb
            print_str += ' повтор с Sn: ' + 'x = ' + str(self.target_locations_x[args.Sn]) + ' y = ' + str(
                self.target_locations_y[args.Sn])

        with open('передатчик_гбн_new.txt', 'a') as f:
            print(print_str, file=f)
        # print(print_str)

    def __init_share_memory(self):
        self.down_plane_points[0] = self.periscope.mirror_down.triangle.point_b.x
        self.down_plane_points[1] = self.periscope.mirror_down.triangle.point_b.y
        self.down_plane_points[2] = self.periscope.mirror_down.triangle.point_b.z
        self.down_plane_points[3] = self.periscope.mirror_down.triangle.point_c.x
        self.down_plane_points[4] = self.periscope.mirror_down.triangle.point_c.y
        self.down_plane_points[5] = self.periscope.mirror_down.triangle.point_c.z

        self.up_plane_points[0] = self.periscope.mirror_up.triangle.point_b.x
        self.up_plane_points[1] = self.periscope.mirror_up.triangle.point_b.y
        self.up_plane_points[2] = self.periscope.mirror_up.triangle.point_b.z
        self.up_plane_points[3] = self.periscope.mirror_up.triangle.point_c.x
        self.up_plane_points[4] = self.periscope.mirror_up.triangle.point_c.y
        self.up_plane_points[5] = self.periscope.mirror_up.triangle.point_c.z

    def __move_target(self, iteration) -> (bool, bool):
        exit_app = False
        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                exit_app = True

        possible_move = ['up', 'down', 'left', 'right']
        key = random.choice(possible_move)
        print(f'target direction: {key}')
        #possible_move = ['up', 'up', 'left', 'left', 'up', 'left', 'up', 'up', 'up', 'left']
        #key = possible_move[iteration]
        delta = 0.03
        self.periscope.prev_target.location.x = float(self.periscope.target.location.x)
        self.periscope.prev_target.location.y = float(self.periscope.target.location.y)

        if key == 'up':
            self.periscope.target.location.y += delta
        elif key == 'down':
            self.periscope.target.location.y -= delta
        elif key == 'right':
            self.periscope.target.location.x += delta
        elif key == 'left':
            self.periscope.target.location.x -= delta

        need_rebuild = True
        return exit_app, need_rebuild

    def run(self):
        self._sender_process.start()
        # self._sender_process = None
        # self._receiver_process.start()

        self.up_plane_process.start()
        self.down_plane_process.start()
        tee = self.periscope.target
        prev_tee = self.periscope.prev_target

        exit_app = False
        iteration = 0
        max_iteration = 10

        while not exit_app and iteration <= max_iteration:

            p1_intersect = self.periscope.laser.intersect_plane(self.periscope.mirror_down.triangle)
            p2_intersect = self.periscope.laser.reflect_plane(self.periscope.mirror_down.triangle). \
                intersect_plane(self.periscope.mirror_up.triangle)
            p_aim = self.periscope.ray_to_aim().intersect_plane(
                Triangle(Point3d(tee.location.x, 0.5, 0.2),
                         Point3d(tee.location.x, 0.4, 0.1),
                         Point3d(tee.location.x, 0.3, 0.5)
                         ))
            self.renderer.render(p1_intersect, p2_intersect, tee, prev_tee, p_aim)

            if iteration == max_iteration:
                sleep(self.sleep_time)
                break

            self.update_log(iteration, p_aim, 'before move target')
            exit_app, need_rebuild = self.__move_target(iteration)

            self.target_locations_x[self.ii] = self.periscope.target.location.x
            self.target_locations_y[self.ii] = self.periscope.target.location.y
            self.ii += 1

            iteration += 1

            self.apply_changes(self.periscope.mirror_down.triangle, self.down_plane_points)
            self.apply_changes(self.periscope.mirror_up.triangle, self.up_plane_points)
            # update log
            sleep(self.sleep_time)

        self.up_plane_process.terminate()
        self.down_plane_process.terminate()
        self._sender_process.terminate()
        self.write_log()
        print(self.count_win)
        print(self.count_all)
        exit()

    def update_log(self, iteration, p_aim, info):
        tee = self.periscope.target.location
        up = self.periscope.mirror_up.triangle
        down = self.periscope.mirror_down.triangle

        output_iteration_list = []
        output_iteration_list.append('-------------info: ' + info + '-------------\n')
        output_iteration_list.append('-------------iteration: ' + str(iteration) + '-------------\n')
        output_iteration_list.append(' '.join(['target: ', str(tee.x), str(tee.y), str(tee.z), '\n']))
        output_iteration_list.append(' '.join(['difference: ', str(p_aim.distance_to_point(tee)), '\n']))
        output_iteration_list.append(
            ' '.join(['up b: ', str(up.point_b.x), str(up.point_b.y), str(up.point_b.z), '\n']))
        output_iteration_list.append(
            ' '.join(['up c: ', str(up.point_c.x), str(up.point_c.y), str(up.point_c.z), '\n']))
        output_iteration_list.append(
            ' '.join(['down b: ', str(down.point_b.x), str(down.point_b.y), str(down.point_b.z), '\n']))
        output_iteration_list.append(
            ' '.join(['down c: ', str(down.point_c.x), str(down.point_c.y), str(down.point_c.z), '\n']))

        diff_target = p_aim.distance_to_point(tee)
        if diff_target < self.periscope.target.radius:
            self.count_win += 1

        self.count_all += 1

        self.log_list.append(''.join(output_iteration_list))

    def write_log(self):
        f = open('/Users/epriimak/Desktop/Diplom/Periscope/logs/a.txt', 'w')
        f.writelines(self.log_list)
        f.close()

    @staticmethod
    def apply_changes(plane: Triangle, arr: mp.Array):
        plane.point_b = Point3d(arr[0], arr[1], arr[2])
        plane.point_c = Point3d(arr[3], arr[4], arr[5])

    @staticmethod
    def final_ray_target_diff(laser: Ray, down_plane: Triangle, up_plane: Triangle, target: Point3d) -> float:
        ray_to_target = laser.reflect_plane(down_plane).reflect_plane(up_plane)
        return target.distance_to_line(ray_to_target.startPos, ray_to_target.startPos + ray_to_target.dir)

    @staticmethod
    def __rotate_plane_in_best_angle(
            periscope: Periscope,
            mirror_loc: MirrorLocation,
            angle_name: Angle,
            step: int
    ):
        angle = Periscope.EPS_ANGLE_DELTA / (2 ** step)
        input_ray = periscope.laser.reflect_plane(periscope.mirror_down.triangle)

        if mirror_loc == MirrorLocation.UP:
            current_plane = periscope.mirror_up.triangle
        else:
            current_plane = periscope.mirror_down.triangle

        input_diff = PeriscopeApplication.final_ray_target_diff(periscope.laser, periscope.mirror_down.triangle,
                                                           periscope.mirror_up.triangle, periscope.target.location)

        plane_angle_plus: Triangle = current_plane.rotate_plane(angle, angle_name)
        plane_angle_minus: Triangle = current_plane.rotate_plane(-angle, angle_name)

        if mirror_loc == MirrorLocation.UP:
            diff_angle_plus = PeriscopeApplication.final_ray_target_diff(periscope.laser, periscope.mirror_down.triangle,
                                                                    plane_angle_plus, periscope.target.location)
            diff_angle_minus = PeriscopeApplication.final_ray_target_diff(periscope.laser, periscope.mirror_down.triangle,
                                                                     plane_angle_minus, periscope.target.location)
        else:
            diff_angle_plus = PeriscopeApplication.final_ray_target_diff(periscope.laser, plane_angle_plus,
                                                                    periscope.mirror_up.triangle,
                                                                    periscope.target.location)
            diff_angle_minus = PeriscopeApplication.final_ray_target_diff(periscope.laser, plane_angle_minus,
                                                                     periscope.mirror_up.triangle,
                                                                     periscope.target.location)

        if math.fabs(diff_angle_plus - diff_angle_minus) < 1e-5:
            return

        if diff_angle_plus < diff_angle_minus:
            diff = diff_angle_plus
            angle_sign = 1
            plane_angle_step = plane_angle_plus
        else:
            diff = diff_angle_minus
            angle_sign = -1
            plane_angle_step = plane_angle_minus

        if mirror_loc == MirrorLocation.UP:
            if not PeriscopeApplication.__check_rotate_relevant(input_ray, plane_angle_step):
                return
        else:
            ray = periscope.laser.reflect_plane(plane_angle_step)
            if not PeriscopeApplication.__check_rotate_relevant(ray, periscope.mirror_up.triangle):
                return  # current_plane.rotate_plane(angle * -angle_sign, angle_name)
            if not PeriscopeApplication.__check_rotate_relevant(periscope.laser, plane_angle_step):
                return

        prev_diff = input_diff
        angle_step = 1
        while diff < prev_diff:
            angle_step += 1
            new_plane_angle_step: Triangle = current_plane.rotate_plane(angle * angle_step * angle_sign, angle_name)
            prev_diff = diff

            if mirror_loc == MirrorLocation.UP:
                if not PeriscopeApplication.__check_rotate_relevant(input_ray, new_plane_angle_step):
                    return
            else:
                ray = periscope.laser.reflect_plane(new_plane_angle_step)
                if not PeriscopeApplication.__check_rotate_relevant(ray, periscope.mirror_up.triangle):
                    return  # current_plane.rotate_plane(angle * -angle_sign, angle_name)
                if not PeriscopeApplication.__check_rotate_relevant(periscope.laser, new_plane_angle_step):
                    return

            plane_angle_step = new_plane_angle_step

            if mirror_loc == MirrorLocation.UP:
                diff = PeriscopeApplication.final_ray_target_diff(periscope.laser, periscope.mirror_down.triangle,
                                                             plane_angle_step, periscope.target.location)
            else:
                diff = PeriscopeApplication.final_ray_target_diff(periscope.laser, plane_angle_step,
                                                             periscope.mirror_up.triangle, periscope.target.location)

        return plane_angle_step

    # if point (on ray and plane) is in triangle
    @staticmethod
    def __check_rotate_relevant(ray: Ray, plane: Triangle) -> bool:
        point_plane_intersect: Point3d = ray.intersect_plane(plane)
        xz_a = Point2d(plane.point_a.x, plane.point_a.z)
        xz_b = Point2d(plane.point_b.x, plane.point_b.z)
        xz_c = Point2d(plane.point_c.x, plane.point_c.z)
        xz_k = Point2d(point_plane_intersect.x, point_plane_intersect.z)

        is_relevant = True
        is_relevant *= PeriscopeApplication.__on_one_side_of_the_plane(Vector2d(xz_b, xz_a), Vector2d(xz_c, xz_a),
                                                                  Vector2d(xz_k, xz_a))
        is_relevant *= PeriscopeApplication.__on_one_side_of_the_plane(Vector2d(xz_c, xz_a), Vector2d(xz_b, xz_a),
                                                                  Vector2d(xz_k, xz_a))
        is_relevant *= PeriscopeApplication.__on_one_side_of_the_plane(Vector2d(xz_b, xz_c), Vector2d(xz_a, xz_c),
                                                                  Vector2d(xz_k, xz_c))
        return is_relevant

    @staticmethod
    def __on_one_side_of_the_plane(v_plane: Vector2d, v2: Vector2d, vk: Vector2d) -> bool:
        pseudo_scalar_v_plane_vk = v_plane.pseudo_scalar_prod(vk)
        pseudo_scalar_v_plane_v2 = v_plane.pseudo_scalar_prod(v2)

        if pseudo_scalar_v_plane_vk * pseudo_scalar_v_plane_v2 > 0:
            return True

        return False

    @staticmethod
    def correct_one_plane(periscope: Periscope, mirror_loc: MirrorLocation, angle: Angle, step: int):
        new_plane = PeriscopeApplication.__rotate_plane_in_best_angle(periscope, mirror_loc, angle, step)
        if new_plane is None:
            return

        if mirror_loc == MirrorLocation.UP:
            periscope.mirror_up.triangle = new_plane
        else:
            periscope.mirror_down.triangle = new_plane

    # implementation for 1 process program
    @staticmethod
    def correct_planes(periscope: Periscope, iteration: int = 0):
        diff = PeriscopeApplication.final_ray_target_diff(periscope.laser, periscope.mirror_down.triangle,
                                                     periscope.mirror_up.triangle, periscope.target.location)

        first_loc_plane = MirrorLocation.UP
        second_loc_plane = MirrorLocation.DOWN
        if iteration % 2 == 0:
            first_loc_plane = MirrorLocation.DOWN
            second_loc_plane = MirrorLocation.UP

        step = 0
        while diff > periscope.target.radius / 2 and step < 10:
            PeriscopeApplication.correct_one_plane(periscope, first_loc_plane, Angle.ROLL, step)
            PeriscopeApplication.correct_one_plane(periscope, first_loc_plane, Angle.PITCH, step)

            PeriscopeApplication.correct_one_plane(periscope, second_loc_plane, Angle.ROLL, step)
            PeriscopeApplication.correct_one_plane(periscope, second_loc_plane, Angle.PITCH, step)

            diff = PeriscopeApplication.final_ray_target_diff(periscope.laser, periscope.mirror_down.triangle,
                                                         periscope.mirror_up.triangle, periscope.target.location)
            step += 1

    @staticmethod
    def find_index(taget: Target, target_locations_x: mp.Array, target_locations_y: mp.Array):
        for i in range(target_locations_x.__sizeof__()):
            if taget.location.x == target_locations_x[i] and \
                    taget.location.y == target_locations_y[i]:
                return i
        return None

    @staticmethod
    def plane_direct_process(self_queue: mp.Queue, arr, periscope: Periscope, plane_loc: MirrorLocation,
                             target_locations_x: mp.Array, target_locations_y: mp.Array,
                             dist_queue: mp.Queue, lose_prob, plane_2_queue: mp.Queue):
        iteration = 0
        Rn = 0

        while True:
            print('up')
            periscope.target = self_queue.get()
            process_name = mp.process.current_process().name
            print_str = process_name + ': принят pkt' + periscope.target.get_description()

            if Rn == PeriscopeApplication.find_index(periscope.target, target_locations_x, target_locations_y):
                print_str += ' доставлен '
                target = Target(init_coords=Point3d(target_locations_x[Rn], target_locations_y[Rn]),
                                radius=0.02)
                PeriscopeApplication._send(dist_queue, target=target, lose_prob=lose_prob)
                print_str += ', send ' + target.get_description()
                plane_2_queue.put(target)
                Rn += 1

                diff = PeriscopeApplication.final_ray_target_diff(periscope.laser, periscope.mirror_down.triangle,
                                                                  periscope.mirror_up.triangle,
                                                                  periscope.target.location)

                first_loc_plane = MirrorLocation.UP
                second_loc_plane = MirrorLocation.DOWN
                if iteration % 2 == 0:
                    first_loc_plane = MirrorLocation.DOWN
                    second_loc_plane = MirrorLocation.UP

                step = 0
                while diff > periscope.target.radius / 2 and step < 10:
                    PeriscopeApplication.correct_one_plane(periscope, first_loc_plane, Angle.ROLL, step)
                    PeriscopeApplication.correct_one_plane(periscope, first_loc_plane, Angle.PITCH, step)

                    PeriscopeApplication.correct_one_plane(periscope, second_loc_plane, Angle.ROLL, step)
                    PeriscopeApplication.correct_one_plane(periscope, second_loc_plane, Angle.PITCH, step)

                    diff = PeriscopeApplication.final_ray_target_diff(periscope.laser, periscope.mirror_down.triangle,
                                                                      periscope.mirror_up.triangle,
                                                                      periscope.target.location)
                    step += 1

                self_plane = periscope.mirror_down.triangle
                if plane_loc == MirrorLocation.UP:
                    self_plane = periscope.mirror_up.triangle

                arr[0] = self_plane.point_b.x
                arr[1] = self_plane.point_b.y
                arr[2] = self_plane.point_b.z
                arr[3] = self_plane.point_c.x
                arr[4] = self_plane.point_c.y
                arr[5] = self_plane.point_c.z
                iteration += 1

            elif Rn < PeriscopeApplication.find_index(periscope.target, target_locations_x, target_locations_y):
                target = Target(init_coords=Point3d(target_locations_x[Rn - 1], target_locations_y[Rn - 1]),
                                radius=0.02)
                PeriscopeApplication._send(dist_queue, target=target, lose_prob=lose_prob)
                print_str += ' send ' + target.get_description()
            else:
                print_str += ' сброшен '
                PeriscopeApplication._send(dist_queue, periscope.target, lose_prob)
                print_str += ' send ' + periscope.target.get_description()

            with open(f'{process_name}_up_process.txt', 'a') as f:
                print(print_str, file=f)


    @staticmethod
    def plane_direct_process_2(self_queue: mp.Queue, arr, periscope: Periscope, plane_loc: MirrorLocation):
        iteration = 0

        while True:
            print('down')
            periscope.target = self_queue.get()
            process_name = mp.process.current_process().name
            str_process = 'process: ' + process_name + ' got: ' + periscope.target.get_description()

            with open(f'{process_name}_down_process.txt', 'a') as f:
                print(str_process, file=f)

            diff = PeriscopeApplication.final_ray_target_diff(periscope.laser, periscope.mirror_down.triangle,
                                                              periscope.mirror_up.triangle,
                                                              periscope.target.location)

            first_loc_plane = MirrorLocation.UP
            second_loc_plane = MirrorLocation.DOWN
            if iteration % 2 == 0:
                first_loc_plane = MirrorLocation.DOWN
                second_loc_plane = MirrorLocation.UP

            step = 0
            while diff > periscope.target.radius / 2 and step < 10:
                PeriscopeApplication.correct_one_plane(periscope, first_loc_plane, Angle.ROLL, step)
                PeriscopeApplication.correct_one_plane(periscope, first_loc_plane, Angle.PITCH, step)

                PeriscopeApplication.correct_one_plane(periscope, second_loc_plane, Angle.ROLL, step)
                PeriscopeApplication.correct_one_plane(periscope, second_loc_plane, Angle.PITCH, step)

                diff = PeriscopeApplication.final_ray_target_diff(periscope.laser, periscope.mirror_down.triangle,
                                                                  periscope.mirror_up.triangle,
                                                                  periscope.target.location)
                step += 1

            self_plane = periscope.mirror_down.triangle
            if plane_loc == MirrorLocation.UP:
                self_plane = periscope.mirror_up.triangle

            arr[0] = self_plane.point_b.x
            arr[1] = self_plane.point_b.y
            arr[2] = self_plane.point_b.z
            arr[3] = self_plane.point_c.x
            arr[4] = self_plane.point_c.y
            arr[5] = self_plane.point_c.z
            iteration += 1


if __name__ == '__main__':
    input_model: str = '2d'
    algorithm: SolveAlgorithm = SolveAlgorithm.DIRECT

    app = PeriscopeApplication(input_model, algorithm)
    app.run()
