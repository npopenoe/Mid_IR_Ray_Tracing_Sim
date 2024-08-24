import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from cassegrain_geo import Point
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_intersection_points(intersections):
    # x, y coordinates of intersections
    x_coords = [point[0] for point in intersections]
    y_coords = [point[1] for point in intersections]

    # Initialize a 1024x1024 pixel array
    image_array = np.zeros((1024, 1024))

    # Calculate the pixel size
    pixel_size_x = 0.027648 / 1024  # detector_width / 1024
    pixel_size_y = 0.027648 / 1024  # detector_height / 1024

    for x, y in zip(x_coords, y_coords):
        # Calculate the pixel indices, shifted to 0 to 1024 range
        i = int((x + 0.027648 / 2) / pixel_size_x)
        j = int((y + 0.027648 / 2) / pixel_size_y)

        if 0 <= i < 1024 and 0 <= j < 1024:
            image_array[i, j] += 1

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(image_array, cmap='inferno', interpolation='nearest', origin='lower', vmin=0, vmax=10)

    # Use make_axes_locatable to create a new axis for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)  # Adjust size and padding as needed

    # Create the colorbar
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Counts', fontsize=18)  # Increase colorbar label font size
    ax.set_xlabel('X (pixels)', fontsize=16)  # Increase x-axis label font size
    ax.set_ylabel('Y (pixels)', fontsize=16)  # Increase y-axis label font size
    ax.set_title('Detector Image', fontsize=20)  # Increase title font size

    '''# Position y-axis label on the right-hand side
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()'''

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.savefig('Detector_image_with_colorbar.svg', transparent=True)
    plt.show()

    # Create a 1D density profile (summing along the y-axis to get the x-axis profile)
    density_profile_x = np.sum(image_array, axis=0)

    # Plot the 1D density profile
    plt.figure(figsize=(10, 4))
    plt.plot(density_profile_x, color='mediumpurple')
    plt.xlabel('X (pixels)', fontsize=16)  # Increase x-axis label font size
    plt.ylabel('Photon Counts', fontsize=18)  # Increase y-axis label font size
    plt.gca().yaxis.set_label_position('right')  # Position y-axis label on the right-hand side
    plt.gca().yaxis.tick_right()  # Position y-axis ticks on the right-hand side
    plt.title('1D Density Profile (X-Axis)', fontsize=20)  # Increase title font size
    plt.grid(True)

    # Increase tick label font size
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.savefig('Density_profile_x.svg', transparent=True)
    plt.show()


def load_from_fits(filename):
    # Load the data from a FITS file
    hdul = fits.open(filename)

    intersections = hdul['INTERSECTIONS'].data
    rays_data = hdul['RAYS'].data

    # Convert ray data back to the original format if necessary
    rays = []
    for row in tqdm(rays_data):
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

    # Plot the data
    plot_intersection_points(loaded_intersections)

    # Print statistics
    '''num_rays_passed = len(loaded_rays)
    num_rays_emitted = total_miss_counter  # Assuming `total_miss_counter` would be stored or recalculated based on rays
    ratio_passed_to_emitted = num_rays_passed / num_rays_emitted if num_rays_emitted > 0 else 0

    print(f'Number of rays that passed through detector: {num_rays_passed}')
    print(f'Number of rays emitted: {num_rays_emitted}')
    print(f'Ratio of number of rays passed through to number of rays emitted: {ratio_passed_to_emitted:.10f}')'''
