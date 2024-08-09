import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from cassegrain_geo import CassegrainGeometry, Point, Hyperbolic, Parabolic
from generate_points import generate_points, AtmosphericModel, layers, number_of_water_molecules, AtmosphericPoint
import time
from tqdm import tqdm

detector_width = 0.027648
detector_height = 0.027648
z_position = 0

def calculate_normal(point, mirror):
    if isinstance(mirror, Parabolic):
        R = mirror.radius_curv
        return np.array([point.x / R, point.y / R, -1])
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

def get_secondary_z(x, y, telescope):
    return surface_secondary(x, y, telescope.secondary.z_position, telescope.secondary.radius_curv, telescope.secondary.K_val)

def bounce_1(ray_direction, normal):
    normal = normal / np.linalg.norm(normal)
    return ray_direction - 2 * np.dot(ray_direction, normal) * normal

def generate_near_parallel_direction(one_radian_deviation=0.01745):
    while True:
        # Generate theta within the small deviation range
        theta = np.random.uniform(0, one_radian_deviation)
        phi = np.random.uniform(0, 2 * np.pi)
        
        # Convert spherical coordinates to Cartesian coordinates
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = -np.cos(theta)

        direction = np.array([dx, dy, dz])

        # Return the direction vector close to parallel
        return direction

def get_layer_wind_speed(z, wind_profile, layers):
    for i in range(len(layers) - 1):
        if layers[i] <= z < layers[i + 1]:
            return wind_profile[i]
    return wind_profile[-1]

def update_points_with_wind(points, time_step, wind_profile, layers, primary_mirror_radius=5.35):
    for point in points:
        wind_speed = get_layer_wind_speed(point.z, wind_profile, layers)
        point.x += wind_speed * time_step

def simulate_wind_effect(cassegrain_geo, test_points, wind_profile, layers, time_step=0.001, primary_mirror_radius=5.35):
    total_miss_counter = 0
    rays = []
    intersections = []
    point_positions = []

    # Continue tracing rays while particles are within the primary mirror
    while True:
        remaining_particles = False
        
        for point in tqdm(test_points):
            # Update the position of the particle based on the wind speed
            wind_speed = get_layer_wind_speed(point.z, wind_profile, layers)
            point.x += wind_speed * time_step

            # Check if the particle is still within the primary mirror's radius
            if np.sqrt(point.x**2 + point.y**2) <= primary_mirror_radius:
                remaining_particles = True
                
                # Trace a new ray for the current position of the particle
                result, miss_counter = trace_ray(cassegrain_geo, point)
                total_miss_counter += miss_counter
                if result is not None:
                    rays.append(result)
                    intersections.append(result[6])
                    point_positions.append((point.x, point.y, point.z))  # Store the point's position

        # If no particles are left within the primary mirror, terminate the simulation
        if not remaining_particles:
            break

    return rays, intersections, total_miss_counter, point_positions


def trace_ray(telescope, point, max_iterations=10000):
    miss_counter = 0
    iteration_counter = 0

    while iteration_counter < max_iterations:

        ray_direction = generate_near_parallel_direction()

        z_primary = surface_primary(point.x, point.y, telescope.primary.radius_curv)
        t_primary = (z_primary - point.z) / ray_direction[2]
        target_primary = Point(
            point.x + t_primary * ray_direction[0],
            point.y + t_primary * ray_direction[1],
            z_primary
        )

        normal_primary = calculate_normal(target_primary, telescope.primary)
        reflected_primary = bounce_1(ray_direction, normal_primary)

        x_secondary = target_primary.x + reflected_primary[0] * (telescope.secondary.z_position - target_primary.z) / reflected_primary[2]
        y_secondary = target_primary.y + reflected_primary[1] * (telescope.secondary.z_position - target_primary.z) / reflected_primary[2]

        if x_secondary**2 + y_secondary**2 <= (0.7090145)**2:
            z_secondary = surface_secondary(x_secondary, y_secondary, telescope.secondary.z_position, telescope.secondary.radius_curv, telescope.secondary.K_val)
            t_secondary = (z_secondary - target_primary.z) / reflected_primary[2]
            target_secondary = Point(
                target_primary.x + t_secondary * reflected_primary[0],
                target_primary.y + t_secondary * reflected_primary[1],
                z_secondary
            )

            normal_secondary = calculate_normal(target_secondary, telescope.secondary)
            reflected_secondary = bounce_1(reflected_primary, normal_secondary)

            t_detector = (z_position - target_secondary.z) / reflected_secondary[2]
            x_detector = target_secondary.x + t_detector * reflected_secondary[0]
            y_detector = target_secondary.y + t_detector * reflected_secondary[1]

            if -detector_width / 2 <= x_detector <= detector_width / 2 and -detector_height / 2 <= y_detector <= detector_height / 2:
                return (point, target_primary, target_secondary, reflected_secondary, normal_primary, normal_secondary, (x_detector, y_detector, z_position)), miss_counter + 1

        miss_counter += 1
        iteration_counter += 1

    return None, miss_counter

def plot_secondary_mirror(ax, telescope):
    r_secondary = 0.7090145
    x_secondary = np.linspace(-r_secondary, r_secondary, 100)
    y_secondary = np.linspace(-r_secondary, r_secondary, 100)
    X_secondary, Y_secondary = np.meshgrid(x_secondary, y_secondary)
    
    def circular_mask(X, Y, radius):
        return np.sqrt(X**2 + Y**2) <= radius
    
    mask_secondary = circular_mask(X_secondary, Y_secondary, r_secondary)
    Z_secondary = np.zeros_like(X_secondary)
    
    for i in range(X_secondary.shape[0]):
        for j in range(X_secondary.shape[1]):
            if mask_secondary[i, j]:
                Z_secondary[i, j] = get_secondary_z(X_secondary[i, j], Y_secondary[i, j], telescope)
            else:
                Z_secondary[i, j] = np.nan
    
    ax.plot_wireframe(X_secondary, Y_secondary, Z_secondary, color='purple', alpha=0.3)

def circular_mask(X, Y, radius):
        return np.sqrt(X**2 + Y**2) <= radius

def visualize_rays(telescope, rays, point_positions, interval=100):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Only call this once, outside the update function
    plot_secondary_mirror(ax, telescope)
    
    theta = np.linspace(0, 2 * np.pi, 100)
    r_primary = 5.4745
    x_primary = np.linspace(-r_primary, r_primary, 100)
    y_primary = np.linspace(-r_primary, r_primary, 100)
    X_primary, Y_primary = np.meshgrid(x_primary, y_primary)
    mask_primary = circular_mask(X_primary, Y_primary, r_primary)
    Z_primary = np.zeros_like(X_primary)
    Z_primary[mask_primary] = (X_primary[mask_primary]**2 + Y_primary[mask_primary]**2) / (2 * telescope.primary.radius_curv)
    Z_primary[~mask_primary] = np.nan
    ax.plot_wireframe(X_primary, Y_primary, Z_primary, color='green', alpha=0.3)

    x_corners = [-detector_width / 2, detector_width / 2, detector_width / 2, -detector_width / 2, -detector_width / 2]
    y_corners = [-detector_height / 2, -detector_height / 2, detector_height / 2, detector_height / 2, -detector_height / 2]
    z_corners = [z_position] * 5

    ax.plot(x_corners, y_corners, z_corners, color='red', label='NIRC2 Detector Plane')
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.set_zlim([-7, 18])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=60)
    ax.set_box_aspect([1, 1, 1])
    
    def update(frame):
        ray_path, pos = rays[frame], point_positions[frame]
        if ray_path is None:
            return

        point, target_primary, target_secondary, reflected_secondary, normal_primary, normal_secondary, detector_intersection = ray_path

        # Plot the rays from the point to the primary mirror
        ax.plot([pos[0], target_primary.x], [pos[1], target_primary.y], [pos[2], target_primary.z], color='orange', linewidth=0.5)
        # Plot the rays from the primary mirror to the secondary mirror
        ax.plot([target_primary.x, target_secondary.x], [target_primary.y, target_secondary.y], [target_primary.z, target_secondary.z], color='steelblue', linewidth=0.5)

        # Plot the extended line after the secondary reflection
        green_line_length = 20.0
        green_endpoint = Point(target_secondary.x + green_line_length * reflected_secondary[0],
                            target_secondary.y + green_line_length * reflected_secondary[1],
                            target_secondary.z + green_line_length * reflected_secondary[2])
        ax.plot([target_secondary.x, green_endpoint.x], [target_secondary.y, green_endpoint.y], [target_secondary.z, green_endpoint.z], color='palevioletred', linewidth=0.5)
        
        # Scatter plot the particle's position
        ax.scatter(pos[0], pos[1], pos[2], color='blue', marker='o', s=2)

    ani = FuncAnimation(fig, update, frames=len(rays), blit=False, interval=interval)
    plt.show()

def plot_intersection_points(intersections):
    x_coords = [point[0] for point in intersections]
    y_coords = [point[1] for point in intersections]

    image_array = np.zeros((1024, 1024))

    pixel_size_x = detector_width / 1024
    pixel_size_y = detector_height / 1024

    for x, y in zip(x_coords, y_coords):
        i = int((x + detector_width / 2) / pixel_size_x)
        j = int((y + detector_height / 2) / pixel_size_y)

        if 0 <= i < 1024 and 0 <= j < 1024:
            image_array[i, j] += 1

    plt.figure(figsize=(8, 8))
    plt.imshow(image_array, cmap='hot', interpolation='nearest', extent=[0, 1024, 0, 1024])
    plt.colorbar(label='Counts')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('Detector Image')
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    F = 149.583
    f1 = 17.489
    b = 2.5
    cassegrain_geo = CassegrainGeometry(F, b, f1)

    atm_model = AtmosphericModel()
    atmospheric_points = generate_points(atm_model, layers, number_of_water_molecules)

    test_points = [AtmosphericPoint(p.x, p.y, p.z, p.temperature, p.humidity, p.emissivity, p.pressure, p.wind, p.cn2) for p in atmospheric_points]

    wind_profile = np.array([16.12590149, 24.75503458, 27.75463769, 29.61640149, 30.96992875, 32.03401622])

    rays, intersections, total_miss_counter, point_positions = simulate_wind_effect(cassegrain_geo, test_points, wind_profile, layers)

    visualize_rays(cassegrain_geo, rays, point_positions)

    num_rays_passed = len(rays)
    num_rays_emitted = total_miss_counter
    ratio_passed_to_emitted = num_rays_passed / num_rays_emitted if num_rays_emitted > 0 else 0

    print(f'Number of rays that passed through detector: {num_rays_passed}')
    print(f'Number of rays emitted: {num_rays_emitted}')
    print(f'Ratio of number of rays passed through to number of rays emitted: {ratio_passed_to_emitted:.10f}')

    plot_intersection_points(intersections)
