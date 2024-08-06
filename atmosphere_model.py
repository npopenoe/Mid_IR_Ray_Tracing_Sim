import numpy as np
import matplotlib.pyplot as plt

class AtmosphericModel:
    def __init__(self):
        self.k_B = 1.38e-23  # Boltzmann constant in J/K
        self.R = 8.314  # Universal gas constant in J/(mol·K)
        self.M_air = 0.029  # Molar mass of Earth's air in kg/mol
        self.M_water = 0.018  # Molar mass of water vapor in kg/mol
        self.N_A = 6.022e23  # Avogadro's number in molecules/mol

    def temperature_profile(self, z):
        T_0 = 15  # Surface temperature in °C
        L = 6.5 / 1000  # Lapse rate in °C/m
        return T_0 - L * (z - 4200)

    def humidity_profile(self, z):
        H_0 = 50  # Surface relative humidity in % at 4200 m
        a = 0.2 / 1000  # Decay constant in 1/m
        return H_0 * np.exp(-a * (z - 4200))

    def emissivity_profile(self, z):
        epsilon_0 = 0.7  # Surface emissivity at 4200 m
        B = 0.1 / 1000  # Rate of decrease in 1/m
        return epsilon_0 * np.exp(-B * (z - 4200))

    def pressure_profile(self, z):
        P_0 = 1012.75  # Surface pressure in hPa at 4200 m
        P_0 = P_0 * 100  # Convert to Pa
        g = 9.81  # Gravity in m/s²
        T_m = self.temperature_profile(z) + 273.15  # Convert to Kelvin
        return P_0 * np.exp(-self.M_air * g * (z - 4200) / (self.R * T_m))

    def wind_profile(self, z):
        W_0 = 16  # Base wind speed at 4200 m in m/s
        a = 5  # Adjusted scale factor for more realistic increase
        b = 1.5 / 1000  # Adjusted rate of increase with altitude in 1/m
        return W_0 + a * np.log1p(b * (z - 4200))

    def cn2_profile(self, z):
        C_n2_0 = 1e-15  # Surface Cn^2 value in m^-2/3 at 4200 m
        d = 0.2 / 1000  # Decay constant in 1/m
        return C_n2_0 * np.exp(-d * (z - 4200))

    def saturation_vapor_pressure(self, T):
        return 0.61121 * np.exp((18.678 - T / 234.5) * (T / (257.14 + T)))

    def calculate_air_density(self, P, T):
        return P * self.M_air / (self.R * T)  # Correct units with universal gas constant

    def calculate_absolute_humidity(self, temperature_values_C, humidity_values, temperature_values_K):
        saturation_vapor_pressure_values = self.saturation_vapor_pressure(temperature_values_C)
        absolute_humidity_values = (humidity_values / 100) * saturation_vapor_pressure_values * self.M_water / (self.k_B * temperature_values_K)
        return absolute_humidity_values

    def calculate_water_vapor_molecules(self, absolute_humidity_values):
        number_of_water_molecules = (absolute_humidity_values * self.N_A) / self.M_water
        return number_of_water_molecules

    def calculate_atmospheric_density(self, pressure_values, temperature_values_K, absolute_humidity_values):
        dry_air_density_values = self.calculate_air_density(pressure_values * (1 - humidity_values / 100), temperature_values_K)
        atmospheric_density_values = dry_air_density_values + absolute_humidity_values
        return atmospheric_density_values

# Define the layers and their altitudes for the troposphere up to 20 km
layers = np.linspace(17, 15800, 6)  # 5 sections from 4200 m to 20000 m in meters
midpoints = (layers[:-1] + layers[1:]) / 2  # Midpoints of the layers

# Initialize the AtmosphericModel class
atm_model = AtmosphericModel()

# Compute the values at the boundary points of the layers
temperature_values = atm_model.temperature_profile(layers)  # In Celsius
temperature_values_K = temperature_values + 273.15  # Convert to Kelvin
humidity_values = atm_model.humidity_profile(layers)
emissivity_values = atm_model.emissivity_profile(layers)
pressure_values = atm_model.pressure_profile(layers)
wind_values = atm_model.wind_profile(layers)
cn2_values = atm_model.cn2_profile(layers)

# Compute air density of dry air
absolute_humidity_values = atm_model.calculate_absolute_humidity(temperature_values, humidity_values, temperature_values_K)
number_of_water_molecules = atm_model.calculate_water_vapor_molecules(absolute_humidity_values)
total_atmospheric_density_values = atm_model.calculate_atmospheric_density(pressure_values, temperature_values_K, absolute_humidity_values)

print(number_of_water_molecules)

# Function to calculate slopes within each layer
def calculate_slope(values, layers):
    slopes = []
    for i in range(len(values) - 1):
        slope = (values[i + 1] - values[i]) / (layers[i + 1] - layers[i])
        slopes.append(slope)
    return slopes

# Calculate slopes for each parameter
temperature_slopes = calculate_slope(temperature_values, layers)
humidity_slopes = calculate_slope(humidity_values, layers)
emissivity_slopes = calculate_slope(emissivity_values, layers)
pressure_slopes = calculate_slope(pressure_values, layers)
wind_slopes = calculate_slope(wind_values, layers)
cn2_slopes = calculate_slope(cn2_values, layers)

# Function to plot the profile
def plot_profile(layers, values, title, xlabel):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(values, layers, 'o', color='blue')
    ax.plot(values, layers, color='blue', linestyle='-')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('log Altitude (m)')  # Altitude in meters
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True)

    # Add layer boundaries as dotted lines
    for i, layer in enumerate(layers):
        if i == 4:  # 5th from the bottom (index 4 in a 0-indexed array)
            ax.axhline(y=layer, color='pink', linestyle='dotted', label='Top of Troposphere')
            ax.axhline(y=layers[0], color='limegreen', linestyle='dotted', label='Mauna Kea Summit')
        else:
            ax.axhline(y=layer, color='paleturquoise', linestyle='dotted')
    
    plt.legend()
    plt.show()

# Plot the profiles
'''plot_profile(layers, absolute_humidity_values, 'Absolute Humidity Profile', 'log Density (kg/m^3)')
plot_profile(layers, number_of_water_molecules, 'Number of Water Vapor Molecules Profile', 'Number of Molecules (molecules/m³)')
plot_profile(layers, total_atmospheric_density_values, 'Atmospheric Density Profile', 'Density (kg/m^3)')
'''