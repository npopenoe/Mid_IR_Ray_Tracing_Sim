import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cassegrain_telescope import CassegrainTelescope, Point, HyperbolicMirror, ParabolicMirror
from generate_points import generate_points, Point

# chooses a specific point on primary mirror
def specific_point_on_primary_mirror(mirror):
    x = 5
    y = 0
    z = (x**2 + y**2) / (4 * telescope.primary.focal_length)
    return Point(x, y, z)

# uses the gradient to calculate the normal vector for both mirrors
def calculate_normal(point, mirror):
    if isinstance(mirror, ParabolicMirror):
        return np.array([2 * point.x / (4 * telescope.primary.focal_length), 2 * point.y / (4 * telescope.primary.focal_length), -1])
    elif isinstance(mirror, HyperbolicMirror):
        return np.array([2 * point.x / telescope.secondary.focal_length, 2 * point.y / telescope.secondary.focal_length, -1])

# uses R = I - 2(I.N)N to calc reflected ray
def reflect(ray_direction, normal):
    normal = normal / np.linalg.norm(normal)
    return ray_direction - 2 * np.dot(ray_direction, normal) * normal

# handles the calculation of bouncing rays
def trace_ray(telescope, point):
    target_primary = specific_point_on_primary_mirror(telescope.primary)

    # direction of the ray from the initial point to the target point on the primary mirror is calculated and normalized
    ray_direction = np.array([target_primary.x - point.x, target_primary.y - point.y, target_primary.z - point.z])
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    # the normal to the primary mirror at the target point is calculated. The ray is then reflected using this normal
    normal_primary = calculate_normal(target_primary, telescope.primary)
    reflected_primary = reflect(ray_direction, normal_primary)

    # intersection of the reflected ray with the secondary mirror is calculated using the z-component of the reflected ray
    t_secondary = telescope.secondary.intersect(target_primary, reflected_primary)
    print(t_secondary,telescope.secondary.position_z, target_primary.z)
    z_secondary = telescope.secondary.position_z + target_primary.z

    target_secondary = Point(target_primary.x + z_secondary * reflected_primary[0],
                             target_primary.y + z_secondary * reflected_primary[1],
                             target_primary.z + z_secondary * reflected_primary[2])
    print(z_secondary)

    # the normal to the secondary mirror at the intersection point is calculated. The ray is then reflected using this normal
    normal_secondary = calculate_normal(target_secondary, telescope.secondary)
    reflected_secondary = reflect_with_double_angle(reflected_primary, normal_secondary)

    # calculates the endpoint of the green line
    green_line_length = 0.5  # Adjust this length as needed
    green_endpoint = Point(target_secondary.x + green_line_length * reflected_secondary[0],
                            target_secondary.y + green_line_length * reflected_secondary[1],
                            target_secondary.z + green_line_length * reflected_secondary[2])

    return point, target_primary, target_secondary, green_endpoint, normal_secondary

# ensures that the normal of the secondary mirror bisects the incoming and outgoing rays
def reflect_with_double_angle(ray_direction, normal):
    normal = normal / np.linalg.norm(normal)
    incidence_angle = np.arccos(np.dot(-ray_direction, normal))
    reflection_angle = 2 * incidence_angle
    rotation_axis = np.cross(ray_direction, normal)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    rotation_matrix = rotation_matrix_around_axis(rotation_axis, reflection_angle)
    reflected_direction = np.dot(rotation_matrix, -normal)
    
    return reflected_direction
def rotation_matrix_around_axis(axis, theta):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def visualize_single_ray_path(telescope, ray_path):
    if ray_path is None:
        print("No successful ray found")
        return

    point, target_primary, target_secondary, green_endpoint, normal_secondary = ray_path

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the initial point
    ax.scatter(point.x, point.y, point.z, color='blue', marker='o', label='Initial Point')

    # Plot the primary reflection
    ax.plot([point.x, target_primary.x], [point.y, target_primary.y], [point.z, target_primary.z], color='orange', label='Primary Reflection')
    ax.plot([target_primary.x, target_secondary.x], [target_primary.y, target_secondary.y], [target_primary.z, target_secondary.z], color='red', label='Secondary Reflection')
    
    # Plot the green line using the angle of reflection
    ax.plot([target_secondary.x, green_endpoint.x], [target_secondary.y, green_endpoint.y], [target_secondary.z, -green_endpoint.z], color='green', label='Reflected Line')
    
    # plot primary focal point
    primary_focal_point = Point(0, 0, telescope.primary.focal_length)
    ax.scatter(primary_focal_point.x, primary_focal_point.y, primary_focal_point.z, color='green', marker='o', label='Primary Focal Point')

    # Plot the primary mirror
    theta = np.linspace(0, 2 * np.pi, 100)
    r = telescope.primary.diameter / 2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    X, Y = np.meshgrid(x, y)
    Z = (X**2 + Y**2) / (4 * telescope.primary.focal_length)
    ax.plot_wireframe(X, Y, Z, color='green', alpha=0.3)
    
    # Plot the secondary mirror
    r_secondary = (telescope.secondary.diameter / 2)
    x_secondary = r_secondary * np.cos(theta) 
    y_secondary = r_secondary * np.sin(theta)
    X_secondary, Y_secondary = np.meshgrid(x_secondary, y_secondary)
    Z_secondary = telescope.secondary.position_z - telescope.secondary.focal_length + np.sqrt(telescope.secondary.focal_length**2 + X_secondary**2 + Y_secondary**2)
    ax.plot_wireframe(X_secondary, Y_secondary, Z_secondary, color='purple', alpha=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=0, azim=-90)
    ax.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1
    #ax.set_xticks(np.arange(-1, 10, 0.5))
    #ax.set_yticks(np.arange(-1, 10, 0.5))
    #ax.set_zticks(np.arange(-1, 10, 0.5))
    plt.legend()
    plt.show()

# test point located directly above mirror
if __name__ == "__main__":
    telescope = CassegrainTelescope(primary_focal_length=17.5, secondary_focal_length=17.5, primary_diameter=10, secondary_diameter=1.4, secondary_position_z=15.35)
    test_point = Point(5, 0, 20)  # Using the specific test point
    ray_path = trace_ray(telescope, test_point)
    visualize_single_ray_path(telescope, ray_path)
