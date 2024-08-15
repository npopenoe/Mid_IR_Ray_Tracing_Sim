import numpy as np
from scipy.integrate import quad

h = 6.626e-34  
c = 3.0e8    
k_B = 1.381e-23 

def planck_law(wavelength, T):
    numerator = 2 * h * c**2 / wavelength**5
    denominator = np.exp(h * c / (wavelength * k_B * T)) - 1
    return numerator / denominator  

# mid ir (4000nm - 5000nm)
lambda_min = 4549e-9
lambda_max = 4790e-9

T = 280

# perform the integration
energy, error = quad(planck_law, lambda_min, lambda_max, args=(T,))
print(f"Total energy emitted over 4000-5000nm: {energy} W·m^-2·sr^-1") # = energy emitted per unit area per unit solid angle per unit time within that wavelength range

