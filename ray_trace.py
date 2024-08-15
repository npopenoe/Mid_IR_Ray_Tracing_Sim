import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cassegrain_geo import CassegrainGeometry, Point, Hyperbolic, Parabolic
from generate_points import generate_points, AtmosphericModel, layers, number_of_water_molecules, AtmosphericPoint
import time
from tqdm import tqdm
from parallelize import parallelize
from astropy.io import fits

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

def generate_random_point_on_primary(X_primary, Y_primary, Z_primary, mask_primary):
    valid_points = np.where(mask_primary)
    idx = np.random.choice(len(valid_points[0]))
    x = X_primary[valid_points][idx]
    y = Y_primary[valid_points][idx]
    z = Z_primary[valid_points][idx]
    return Point(x, y, z)

def calculate_solid_angle(x, y, z, mirror_radius):
    # Define the integrand for off-axis points
    def integrand(r_prime, phi):
        r = np.sqrt((x - r_prime * np.cos(phi))**2 + (y - r_prime * np.sin(phi))**2 + z**2)
        cos_theta = z / r
        return (r_prime * cos_theta) / (r**2)
    
    # Integration limits
    r_prime_max = mirror_radius
    phi_max = 2 * np.pi
    
    # Discretize the integrand over a grid
    r_primes = np.linspace(0, r_prime_max, 100)
    phis = np.linspace(0, phi_max, 100)
    dr_prime = r_prime_max / 100
    dphi = phi_max / 100
    
    # Perform the double integral
    solid_angle = 0
    for r_prime in r_primes:
        for phi in phis:
            solid_angle += integrand(r_prime, phi) * dr_prime * dphi
    
    return solid_angle

# Define the physical dimensions of the detector in meters
detector_width = 0.027648
detector_height = 0.027648
z_detector = -2.5 

def generate_random_direction_within_solid_angle(solid_angle):
    # Calculate the maximum polar angle theta_max based on the solid angle
    theta_max = np.arccos(1 - solid_angle / (2 * np.pi))

    # Seed the random number generator
    np.random.seed(int(time.time() * 1000) % 2**32)

    while True:
        # Generate a random direction on the unit hemisphere
        cos_theta = np.random.uniform(np.cos(theta_max), 1)
        theta = np.arccos(cos_theta)
        phi = np.random.uniform(0, 2 * np.pi)

        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = -np.cos(theta)

        direction = np.array([dx, dy, dz])

        return direction
    
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

def trace_ray(telescope, point, max_iterations=1000):
    miss_counter = 0
    iteration_counter = 0

    while iteration_counter < max_iterations:
        # Generate a near-parallel direction within the small deviation range
        ray_direction = generate_near_parallel_direction()

        # Ensure the ray intersects with the primary mirror surface
        z_primary = surface_primary(point.x, point.y, telescope.primary.radius_curv)
        t_primary = (z_primary - point.z) / ray_direction[2]
        target_primary = Point(
            point.x + t_primary * ray_direction[0],
            point.y + t_primary * ray_direction[1],
            z_primary
        )

        # Calculate the normal to the primary mirror at the target point
        normal_primary = calculate_normal(target_primary, telescope.primary)
        reflected_primary = bounce_1(ray_direction, normal_primary)

        # Calculate intersection with the secondary mirror
        x_secondary = target_primary.x + reflected_primary[0] * (telescope.secondary.z_position - target_primary.z) / reflected_primary[2]
        y_secondary = target_primary.y + reflected_primary[1] * (telescope.secondary.z_position - target_primary.z) / reflected_primary[2]

        # Check if the intersection point is within the secondary mirror's bounds
        if x_secondary**2 + y_secondary**2 <= (0.7090145)**2:
            z_secondary = surface_secondary(x_secondary, y_secondary, telescope.secondary.z_position, telescope.secondary.radius_curv, telescope.secondary.K_val)
            t_secondary = (z_secondary - target_primary.z) / reflected_primary[2]
            target_secondary = Point(
                target_primary.x + t_secondary * reflected_primary[0],
                target_primary.y + t_secondary * reflected_primary[1],
                z_secondary
            )

            # Calculate the normal to the secondary mirror at the intersection point
            normal_secondary = calculate_normal(target_secondary, telescope.secondary)
            reflected_secondary = bounce_1(reflected_primary, normal_secondary)

            # Check if the reflected ray intersects the detector plane within its bounds
            t_detector = (z_detector - target_secondary.z) / reflected_secondary[2]
            x_detector = target_secondary.x + t_detector * reflected_secondary[0]
            y_detector = target_secondary.y + t_detector * reflected_secondary[1]

            if -detector_width / 2 <= x_detector <= detector_width / 2 and -detector_height / 2 <= y_detector <= detector_height / 2:
                '''print(f'Number of misses before hitting the secondary mirror: {miss_counter}')
                print(f'Ray intersects the detector at (x, y, z): ({x_detector}, {y_detector}, {z_detector})')'''
                return (point, target_primary, target_secondary, reflected_secondary, normal_primary, normal_secondary, (x_detector, y_detector, z_detector)), miss_counter + 1

        miss_counter += 1
        iteration_counter += 1

    '''print(f'No valid ray found after {max_iterations} iterations')'''
    return None, miss_counter

'''def plot_secondary_mirror(ax, telescope):
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

        point, target_primary, target_secondary, reflected_secondary, normal_primary, normal_secondary, detector_intersection = ray_path

        # Plot the initial point
        ax.scatter(point.x, point.y, point.z, color='blue', marker='o', s = 2, label='Atmospheric Particle')

        # Plot the primary reflection
        ax.plot([point.x, target_primary.x], [point.y, target_primary.y], [point.z, target_primary.z], color='orange', linewidth = 0.5, label='Photon Ray')
        ax.plot([target_primary.x, target_secondary.x], [target_primary.y, target_secondary.y], [target_primary.z, target_secondary.z], linewidth = 0.5, color='steelblue', label='Primary Reflection')

        # Plot the green line using the angle of reflection
        green_line_length = 20.0  # Adjust this length as needed
        green_endpoint = Point(target_secondary.x + green_line_length * reflected_secondary[0],
                               target_secondary.y + green_line_length * reflected_secondary[1],
                               target_secondary.z + green_line_length * reflected_secondary[2])
        ax.plot([target_secondary.x, green_endpoint.x], [target_secondary.y, green_endpoint.y], [target_secondary.z, green_endpoint.z], color='palevioletred', linewidth = 0.5, label='Secondary Reflection')

    # Create a mask for the circular mirrors
    def circular_mask(X, Y, radius):
        return np.sqrt(X**2 + Y**2) <= radius

    # Plot the primary mirror with a circular mask
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

    # Define the corners of the detector in the XY plane
    x_corners = [-detector_width / 2, detector_width / 2, detector_width / 2, -detector_width / 2, -detector_width / 2]
    y_corners = [-detector_height / 2, -detector_height / 2, detector_height / 2, detector_height / 2, -detector_height / 2]
    z_corners = [z_detector] * 5

    # Plot the detector plane
    ax.plot(x_corners, y_corners, z_corners, color = 'red', label='NIRC2 Detector Plane')

    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.set_zlim([-7, 18])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=60)
    ax.set_box_aspect([1, 1, 1])
    #plt.show()'''

def plot_intersection_points(intersections):
    # x, y coordinates of intersections
    x_coords = [point[0] for point in intersections]
    y_coords = [point[1] for point in intersections]

    # Initialize a 1024x1024 pixel array
    image_array = np.zeros((1024, 1024))

    # Calculate the pixel size
    pixel_size_x = detector_width / 1024
    pixel_size_y = detector_height / 1024

    for x, y in zip(x_coords, y_coords):
        # Calculate the pixel indices, shifted to 0 to 1024 range
        i = int((x + detector_width / 2) / pixel_size_x)
        j = int((y + detector_height / 2) / pixel_size_y)

        if 0 <= i < 1024 and 0 <= j < 1024:
            image_array[i, j] += 1

    # Plot the image with pixel coordinates
    plt.figure(figsize=(10, 10))
    plt.imshow(image_array, cmap = 'inferno', interpolation='nearest', origin = 'lower', vmin = 0, vmax = 15)
    plt.colorbar(label='Counts')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('Detector Image')
    plt.savefig('Detector_image.svg', transparent=True)
    plt.show()

def save_to_fits(filename, intersections, rays):
    # Create a primary HDU
    primary_hdu = fits.PrimaryHDU()

    # Convert intersections and rays data to arrays for FITS
    intersection_array = np.array(intersections)
    
    # Define the data type for the ray array to match the structure
    dtype = np.dtype([
        ('atm_x', np.float64), ('atm_y', np.float64), ('atm_z', np.float64),
        ('prim_x', np.float64), ('prim_y', np.float64), ('prim_z', np.float64),
        ('sec_x', np.float64), ('sec_y', np.float64), ('sec_z', np.float64),
        ('ref_sec_x', np.float64), ('ref_sec_y', np.float64), ('ref_sec_z', np.float64),
        ('norm_prim_x', np.float64), ('norm_prim_y', np.float64), ('norm_prim_z', np.float64),
        ('norm_sec_x', np.float64), ('norm_sec_y', np.float64), ('norm_sec_z', np.float64),
        ('det_x', np.float64), ('det_y', np.float64), ('det_z', np.float64)
    ])
    
    # Populate the ray array
    ray_array = np.array([(
        ray[0].x, ray[0].y, ray[0].z,  # Atmospheric Point
        ray[1].x, ray[1].y, ray[1].z,  # Target Primary
        ray[2].x, ray[2].y, ray[2].z,  # Target Secondary
        ray[3][0], ray[3][1], ray[3][2],  # Reflected Secondary
        ray[4][0], ray[4][1], ray[4][2],  # Normal Primary
        ray[5][0], ray[5][1], ray[5][2],  # Normal Secondary
        ray[6][0], ray[6][1], ray[6][2]   # Detector Intersection
    ) for ray in rays], dtype=dtype)

    # Create HDUs for intersections and rays
    intersections_hdu = fits.ImageHDU(data=intersection_array, name='INTERSECTIONS')
    rays_hdu = fits.BinTableHDU(data=ray_array, name='RAYS')

    # Combine the HDUs into an HDUList
    hdul = fits.HDUList([primary_hdu, intersections_hdu, rays_hdu])

    # Write to a FITS file
    hdul.writeto(filename, overwrite=True)

def load_from_fits(filename):
    # Load the data from a FITS file
    hdul = fits.open(filename)

    intersections = hdul['INTERSECTIONS'].data
    rays_data = hdul['RAYS'].data

    # Convert ray data back to the original format if necessary
    rays = []
    for row in rays_data:
        atmospheric_point = Point(row['atm_x'], row['atm_y'], row['atm_z'])
        primary_target = Point(row['prim_x'], row['prim_y'], row['prim_z'])
        secondary_target = Point(row['sec_x'], row['sec_y'], row['sec_z'])
        reflected_secondary = np.array([row['ref_sec_x'], row['ref_sec_y'], row['ref_sec_z']])
        normal_primary = np.array([row['norm_prim_x'], row['norm_prim_y'], row['norm_prim_z']])
        normal_secondary = np.array([row['norm_sec_x'], row['norm_sec_y'], row['norm_sec_z']])
        detector_intersection = (row['det_x'], row['det_y'], row['det_z'])
        
        rays.append((
            atmospheric_point, 
            primary_target, 
            secondary_target, 
            reflected_secondary, 
            normal_primary, 
            normal_secondary, 
            detector_intersection
        ))

    hdul.close()

    return intersections, rays

# Skip the ray tracing part if you only want to plot the existing data
if __name__ == "__main__":    
    # Load the data from the FITS file
    loaded_intersections, loaded_rays = load_from_fits('ray_trace_results.fits')

    # Plot the data with the inferno colormap
    plot_intersection_points(loaded_intersections)



'''if __name__ == "__main__":
    F = 149.583
    f1 = 17.489
    b = 2.5
    cassegrain_geo = CassegrainGeometry(F, b, f1)
    
    atm_model = AtmosphericModel()
    atmospheric_points = generate_points(atm_model, layers, number_of_water_molecules)

    # Use the generated atmospheric points for the test
    test_points = [AtmosphericPoint(p.x, p.y, p.z, p.temperature, p.humidity, p.emissivity, p.pressure, p.wind, p.cn2) for p in atmospheric_points]

    
    total_miss_counter = 0
    rays = []
    intersections = []
    
    for point in tqdm(test_points):
        result, miss_counter = trace_ray(cassegrain_geo, point)
        total_miss_counter += miss_counter
        if result is not None:
            rays.append(result)
            intersections.append(result[6])  # Append the intersection point
    visualize_rays(cassegrain_geo, rays)
    
    
    @parallelize(use_tqdm = True)
    def compute_ray_trace(point):
        result, miss_counter = trace_ray(cassegrain_geo, point)
        return result, miss_counter
    parallel_result = compute_ray_trace(test_points)

    for result, miss_counter in parallel_result:
        total_miss_counter += miss_counter
        if result is not None:
            rays.append(result)
            intersections.append(result[6])
    #visualize_rays(cassegrain_geo, rays)

    # Print statistics
    num_rays_passed = len(rays)
    num_rays_emitted = total_miss_counter
    ratio_passed_to_emitted = num_rays_passed / num_rays_emitted if num_rays_emitted > 0 else 0

    print(f'Number of rays that passed through detector: {num_rays_passed}')
    print(f'Number of rays emitted: {num_rays_emitted}')
    print(f'Ratio of number of rays passed through to number of rays emitted: {ratio_passed_to_emitted:.10f}')

    # Save the data to a FITS file after simulation
    save_to_fits('ray_trace_results.fits', intersections, rays)

    # Load the data later
    loaded_intersections, loaded_rays = load_from_fits('ray_trace_results.fits')

    # Use the loaded data for plotting or further analysis
    plot_intersection_points(loaded_intersections)
'''

    

