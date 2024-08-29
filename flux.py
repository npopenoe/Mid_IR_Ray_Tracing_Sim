import numpy as np
from scipy.integrate import quad
from atmosphere_model import AtmosphericModel
from generate_points import AtmosphericModel

# constants
h = 6.626e-34       # Planck's constant (J·s)
c = 3.0e8           # Speed of light (m/s)
k_B = 1.381e-23     # Boltzmann's constant (J/K)
s = 2.375e-7        # meters
lambda_min = 4549e-9  # meters
lambda_max = 4790e-9  # meters
avg_lambda = 4695e-9 # meters
T = 280 # in Kelvin
rays_emitted = 56492000 # got these nums from running the ray_trace for 1211758 molecules and just copied in
total_intersections = 294000
number_molec = 1211758

def planck_law(wavelength, T):
    numerator = 2 * h * c**2 / wavelength**5
    denominator = np.exp(h * c / (wavelength * k_B * T)) - 1
    return numerator / denominator  # W·m^-2·sr^-1·m^-1

# EQUATIONS

spectral_radiance, error = quad(planck_law, lambda_min, lambda_max, args=(T,)) # integration over the specified wavelength range
atm_model = AtmosphericModel() # initializes the atmospheric model and generates points
molecule_SA = 4 * np.pi * s**2  # m^2 -- # SA of a water molecule 4pis^2
effective_area = number_molec * molecule_SA  # number of molecules in atm multiply by surface area
ratio_rays_etod = total_intersections/rays_emitted # equals 0.005 
total_energy = spectral_radiance * effective_area * 1/4*np.pi # in Watts
energy_per_ray = total_energy / rays_emitted
energy_received_by_detector = energy_per_ray * total_intersections
energy_per_photon = h * c / avg_lambda
photons_per_ray = energy_per_ray / energy_per_photon 
photons_detected = photons_per_ray * total_intersections
energy_detected = photons_detected * energy_per_photon
photons_detected_per_second_per_pixel = (energy_detected / energy_per_photon ) / (1024*1024)


# all the print statements 

print(f"Spectral radiance over {lambda_min*1e9:.0f}-{lambda_max*1e9:.0f}nm: {spectral_radiance} W·m^-2·sr^-1")
print(f"number_of_water_molecules: {1211758}")
print(f"molecule_SA: {molecule_SA}")
print(f"effective area: {effective_area}")
print(f"ratio_rays_emit_to_detect: {ratio_rays_etod}")
print(f"Total energy emitted over {lambda_min*1e9:.0f}-{lambda_max*1e9:.0f}nm: {total_energy} W")
print(f"Energy per ray: {energy_per_ray} J")
print(f"Energy received by detector: {energy_received_by_detector} J/s")
print(f"Energy per photon: {energy_per_photon} J")
print(f"photons per ray:{photons_per_ray}") 
print(f"photons detected: {photons_detected}")
print(f"energy detected: {energy_detected}")
print(f"Photons detected per second: {photons_detected_per_second_per_pixel:.3e} photons/s")



