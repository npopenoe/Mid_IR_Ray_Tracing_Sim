import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cassegrain_geo import CassegrainGeometry, Point, Hyperbolic, Parabolic

def calculate_normal(point, mirror):
    if isinstance(mirror, Parabolic):
        R = mirror.radius_curv
        return np.array([point.x / R,-point.y / R, -1])
    elif isinstance(mirror, Hyperbolic):
        R = mirror.radius_curv
        K = mirror.K_val
        return np.array([
            point.x / np.sqrt(R**2 - (K + 1) * (point.x**2 + point.y**2)), 
            point.y / np.sqrt(R**2 - (K + 1) * (point.x**2 + point.y**2)), 
            -1
        ])

def surface_primary(x, y, primary_radius_curv):
    return (x**2 + y**2) / (2 * primary_radius_curv)

def surface_secondary(x, y, secondary_z_position, secondary_radius_curv, secondary_K):
    z = secondary_z_position + (secondary_radius_curv - np.sqrt(secondary_radius_curv**2 - (secondary_K + 1) * (x**2 + y**2))) / (secondary_K + 1)
    return z

# gets z-value on the secondary mirror
def get_secondary_z(x, y, telescope):
    return surface_secondary(x, y, telescope.secondary_z_position, telescope.secondary_radius_curv, telescope.secondary_K)

# uses R = I - 2(I.N)N to calc reflected ray
def bounce_1(ray_direction, normal):
    normal = normal / np.linalg.norm(normal)
    return ray_direction - 2 * np.dot(ray_direction, normal) * normal

def trace_ray(telescope, point):
    # calc the direction of the ray to the primary mirror (vertical downward)
    ray_direction = np.array([0, 0, -1])
    
    # finds the intersection with the primary mirror
    z_primary = surface_primary(point.x, point.y, telescope.primary.radius_curv)  # primary mirror surface
    t_primary = (z_primary - point.z) / ray_direction[2]  # calc the distance to the intersection
    target_primary = Point(
        point.x + t_primary * ray_direction[0],
        point.y + t_primary * ray_direction[1],
        z_primary
    )
    
    # calc the normal to the primary mirror at the target point
    normal_primary = calculate_normal(target_primary, telescope.primary)
    reflected_primary = bounce_1(ray_direction, normal_primary)
    
    # calc intersection with the secondary mirror
    x_secondary = target_primary.x + reflected_primary[0] * (telescope.secondary_z_position - target_primary.z) / reflected_primary[2]
    y_secondary = target_primary.y + reflected_primary[1] * (telescope.secondary_z_position - target_primary.z) / reflected_primary[2]
    z_secondary = surface_secondary(x_secondary, y_secondary, telescope.secondary_z_position, telescope.secondary_radius_curv, telescope.secondary_K)
    
    t_secondary = (z_secondary - target_primary.z) / reflected_primary[2]  # calc the distance to the intersection on the secondary mirror
    target_secondary = Point(
        target_primary.x + t_secondary * reflected_primary[0],
        target_primary.y + t_secondary * reflected_primary[1],
        z_secondary
    )
    
    # calc the normal to the secondary mirror at the intersection point
    normal_secondary = calculate_normal(target_secondary, telescope.secondary)
    reflected_secondary = bounce_1(reflected_primary, normal_secondary)

    return point, target_primary, target_secondary, reflected_secondary, normal_primary, normal_secondary

def plot_secondary_mirror(ax, telescope):
    # defines the radius of the secondary mirror
    r_secondary = 0.7090145  # Actual radius of the secondary mirror
    x_secondary = np.linspace(-r_secondary, r_secondary, 100)
    y_secondary = np.linspace(-r_secondary, r_secondary, 100)
    X_secondary, Y_secondary = np.meshgrid(x_secondary, y_secondary)
    
    def circular_mask(X, Y, radius):
        return np.sqrt(X**2 + Y**2) <= radius
    
    # apply circular mask to confine plotting area within secondary mirror's radius
    mask_secondary = circular_mask(X_secondary, Y_secondary, r_secondary)
    Z_secondary = np.zeros_like(X_secondary)
    
    # z values for the secondary mirror's surface using get_secondary_z
    for i in range(X_secondary.shape[0]):
        for j in range(X_secondary.shape[1]):
            if mask_secondary[i, j]:
                Z_secondary[i, j] = get_secondary_z(X_secondary[i, j], Y_secondary[i, j], telescope)
            else:
                Z_secondary[i, j] = np.nan  # Points outside the mirror are set to NaN
    
    # wireframe for the secondary mirror
    ax.plot_wireframe(X_secondary, Y_secondary, Z_secondary, color='purple', alpha=0.3)

def visualize_rays(telescope, rays):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for ray_path in rays:
        if ray_path is None:
            continue

        point, target_primary, target_secondary, reflected_secondary, normal_primary, normal_secondary = ray_path

        # plot initial point
        ax.scatter(point.x, point.y, point.z, color='blue', marker='o', label='Atmospheric Particle')

        # plot primary reflection
        ax.plot([point.x, target_primary.x], [point.y, target_primary.y], [point.z, target_primary.z], color='orange', label='Photon Ray')
        ax.plot([target_primary.x, target_secondary.x], [target_primary.y, target_secondary.y], [target_primary.z, target_secondary.z], color='steelblue', label='Primary Reflection')

        '''     # plotting normal vector at the mirror intersection
        ax.quiver(target_secondary.x, target_secondary.y, target_secondary.z, 
                  normal_secondary[0], normal_secondary[1], normal_secondary[2], 
                  length=1, color='red', normalize=True, arrow_length_ratio=0.1)
        
        ax.quiver(target_primary.x, target_primary.y, target_primary.z, 
                  -normal_primary[0], -normal_primary[1], -normal_primary[2], 
                  length=1, color='blue', normalize=True, arrow_length_ratio=0.1)'''

        # plot the final line using the angle of reflection
        pink_line_length = 20.0 
        pink_endpoint = Point(target_secondary.x + pink_line_length * reflected_secondary[0],
                               target_secondary.y + pink_line_length * reflected_secondary[1],
                               target_secondary.z + pink_line_length * reflected_secondary[2])
        ax.plot([target_secondary.x, pink_endpoint.x], [target_secondary.y, pink_endpoint.y], [target_secondary.z, pink_endpoint.z], color='palevioletred', label='Secondary Reflection')

    # mask for the circular mirrors
    def circular_mask(X, Y, radius):
        return np.sqrt(X**2 + Y**2) <= radius

    # plot the mirrors
    theta = np.linspace(0, 2 * np.pi, 100)
    r_primary = 5.4745  # Actual radius of the primary mirror
    x_primary = np.linspace(-r_primary, r_primary, 100)
    y_primary = np.linspace(-r_primary, r_primary, 100)
    X_primary, Y_primary = np.meshgrid(x_primary, y_primary)
    mask_primary = circular_mask(X_primary, Y_primary, r_primary)
    Z_primary = np.zeros_like(X_primary)
    Z_primary[mask_primary] = (X_primary[mask_primary]**2 + Y_primary[mask_primary]**2) / (2 * telescope.primary.radius_curv)
    Z_primary[~mask_primary] = np.nan
    ax.plot_wireframe(X_primary, Y_primary, Z_primary, color='green', alpha=0.3)
    plot_secondary_mirror(ax, telescope)

    # Plot focal points for reference
    ax.scatter(0, 0, 17.5, color='green', marker='o', label='Primary Focal Point', alpha=0.3)
    ax.scatter(0, 0, -2.5, color='purple', marker='o', label='Back Focal Dist', alpha=0.3) 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=60)
    ax.set_box_aspect([1, 1, 1])
    plt.show()

if __name__ == "__main__":
    F = 149.583
    f1 = 17.489
    b = 2.5
    cassegrain_geo = CassegrainGeometry(F, b, f1)
    
    # points across the diameter of the primary mirror spaced by 0.1
    x_values = np.linspace(-5, 5, 11)
    test_points = [Point(x, 0, 20) for x in x_values]

    rays = [trace_ray(cassegrain_geo, point) for point in test_points]
    visualize_rays(cassegrain_geo, rays)