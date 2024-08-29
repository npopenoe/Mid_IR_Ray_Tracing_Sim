import numpy as np
from scipy.integrate import quad
from atmosphere_model import AtmosphericModel, layers, total_atmospheric_density_values
from ray_trace import plot_intersection_points
from generate_points import generate_points, AtmosphericModel

# Constants
h = 6.626e-34  # Planck's constant (J·s)
c = 3.0e8      # Speed of light (m/s)
k_B = 1.381e-23  # Boltzmann's constant (J/K)

def planck_law(wavelength, T):
    numerator = 2 * h * c**2 / wavelength**5
    denominator = np.exp(h * c / (wavelength * k_B * T)) - 1
    return numerator / denominator  # W·m^-2·sr^-1·m^-1

# Wavelength range in meters (for mid-IR)
lambda_min = 4549e-9  # meters
lambda_max = 4790e-9  # meters
T = 280  # Temperature in Kelvin

# Perform the integration over the specified wavelength range
spectral_radiance, error = quad(planck_law, lambda_min, lambda_max, args=(T,))
print(f"Spectral radiance over {lambda_min*1e9:.0f}-{lambda_max*1e9:.0f}nm: {spectral_radiance} W·m^-2·sr^-1")

# Initialize the atmospheric model and generate points
atm_model = AtmosphericModel()
scaling_factor = 2e46  # Adjust this factor as needed

# Calculate total number of particles across all layers
print(f"number_of_water_molecules:{1211758}")

# SA of a water molecule (assuming radius = 1 Å)
molecule_SA = (4) * np.pi * (2.375e-7)**2  # m^2
print(f"molecule_SA: {molecule_SA}")

# Calculate effective area subtended by water vapor molecules
effective_area = 1211758 * molecule_SA  # Total effective area in m^2
print(f"effective area: {effective_area}")

# Plot intersections and get total number
rays_emitted = 56492000
total_intersections = 294000
ratio_rays_etod = total_intersections/rays_emitted
print(f"ratio_rays_emit_to_detect:{ratio_rays_etod}")

# Calculate total energy and energy per ray
total_energy = spectral_radiance * effective_area * 1/4*np.pi # W

# Divide by the total number of intersections to get energy per ray
energy_per_ray = total_energy / rays_emitted

print(f"Total energy emitted over {lambda_min*1e9:.0f}-{lambda_max*1e9:.0f}nm: {total_energy} W")
print(f"Energy per ray: {energy_per_ray} J")

energy_received_by_detector = energy_per_ray * total_intersections
print(f"Energy received by detector: {energy_received_by_detector} J/s")

avg_lambda = 46695e-9
# energy per photon calculation
energy_per_photon = h * c / avg_lambda
print(f"Energy per photon: {energy_per_photon} J")

photons_per_ray = energy_per_ray / energy_per_photon 
print(f"photons per ray:{photons_per_ray}") 

photons_detected = photons_per_ray * total_intersections
print(f"photons detected: {photons_detected}")

energy_detected = photons_detected * energy_per_photon
print(f"energy detected: {energy_detected}")
      
# photons per second
photons_detected_per_second_per_pixel = (energy_detected / energy_per_photon ) / (1024*1024)
print(f"Photons detected per second: {photons_detected_per_second_per_pixel:.3e} photons/s")



