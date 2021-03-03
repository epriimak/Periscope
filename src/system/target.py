from src.geometry import Point3d

class Target:
    def __init__(self, init_coords: Point3d,  radius: float):
        self.location: Point3d = init_coords
        self.radius = radius

    def __str__(self) -> str:
        return f'Coords: x: {self.location.x} y: {self.location.y} z: {self.location.z}'

    def get_description(self):
        x = format(self.location.x, '.3f')
        y = format(self.location.y, '.3f')
        return f'Coords is: x: {x} y: {y}'





