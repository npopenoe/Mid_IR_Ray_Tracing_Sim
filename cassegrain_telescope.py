import numpy as np

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z})"

class ParabolicMirror:
    def __init__(self, focal_length, diameter, z_position):
        self.focal_length = focal_length
        self.diameter = diameter
        self.z_position = z_position

    def reflect(self, point):
        x, y = point.x, point.y
        z = (x**2 + y**2) / (4 * self.focal_length)
        return Point(x, y, z)
    

class HyperbolicMirror:
    def __init__(self, focal_length, diameter, position_z):
        self.focal_length = focal_length
        self.diameter = diameter
        self.position_z = position_z
    
    def intersect(self, point, direction):
        # Define the hyperbolic mirror equation parameters
        a = direction[0]**2 + direction[1]**2 - direction[2]**2
        b = 2 * (point.x * direction[0] + point.y * direction[1] - (self.position_z - self.focal_length) * direction[2])
        c = point.x**2 + point.y**2 - (self.position_z - self.focal_length)**2

        discriminant = b**2 - 4 * a * c

        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        t = max(t1, t2)
        return t2
        
    def calculate_normal(self, point):
        return np.array([2 * point.x / self.focal_length, 2 * point.y / self.focal_length, -1])
    
# uses R = I - 2(I.N)N to calc reflected ray
def reflect(ray_direction, normal):
    return ray_direction - 2 * np.dot(ray_direction, normal) * normal


class CassegrainTelescope:
    def __init__(self, primary_focal_length, secondary_focal_length, primary_diameter, secondary_diameter, secondary_position_z):
        self.primary = ParabolicMirror(primary_focal_length, primary_diameter, 0)
        self.secondary = HyperbolicMirror(secondary_focal_length, secondary_diameter, secondary_position_z)

    def trace_ray(self, point):
        reflected_primary = self.primary.reflect(point)
        reflected_secondary = self.secondary.reflect(reflected_primary)
        return reflected_secondary
        