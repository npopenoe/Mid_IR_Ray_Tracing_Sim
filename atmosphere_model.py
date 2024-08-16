import numpy as np
import matplotlib.pyplot as plt

class AtmosphericModel:
    def __init__(self):
        self.k_B = 1.38e-23  # Boltzmann constant in J/K
        self.M_air = 4.8e-26  #  mass of Earth's air in kg
        self.M_water = 3e-27  # mass of water vapor in kg
        self.N_A = 6.022e23  # Avogadro's number in molecules/mol

    def temperature_profile(self, z):
        T_0 = 15  # Surface temperature in °C
        L = 6.5 / 1000  # Lapse rate in °C/m
        return T_0 - L * (z)

    def humidity_profile(self, z):
        H_0 = 50  # Surface relative humidity in % at 4200 m
        a = 0.2 / 1000  # Decay constant in 1/m
        return H_0 * np.exp(-a * (z))

    def emissivity_profile(self, z):
        epsilon_0 = 0.7  # Surface emissivity at 4200 m
        B = 0.1 / 1000  # Rate of decrease in 1/m
        return epsilon_0 * np.exp(-B * (z))

    def pressure_profile(self, z):
        P_0 = 1012.75  # Surface pressure in hPa at 4200 m
        P_0 = P_0 * 100  # Convert to Pa
        g = 9.81  # Gravity in m/s²
        T_m = self.temperature_profile(z) + 273.15  # Convert to Kelvin
        return P_0 * np.exp(-self.M_air * g * (z) / (self.k_B * T_m))

    def wind_profile(self, z):
        W_0 = 16  # Base wind speed at 4200 m in m/s
        a = 5  # Adjusted scale factor for more realistic increase
        b = 1.5 / 1000  # Adjusted rate of increase with altitude in 1/m
        return W_0 + a * np.log1p(b * (z))

    def cn2_profile(self, z):
        C_n2_0 = 1e-15  # Surface Cn^2 value in m^-2/3 at 4200 m
        d = 0.2 / 1000  # Decay constant in 1/m
        return C_n2_0 * np.exp(-d * (z))

    def saturation_vapor_pressure(self, T):
        return 0.61121 * np.exp((18.678 - T / 234.5) * (T / (257.14 + T)))

    def calculate_air_density(self, P, T):
        return P * self.M_air / (self.k_B * T)  # Correct units with universal gas constant

    def calculate_absolute_humidity(self, temperature_values_C, humidity_values, temperature_values_K):
        saturation_vapor_pressure_values = 0.61121 * np.exp((18.678 - temperature_values_C / 234.5) * (temperature_values_C / (257.14 + temperature_values_C)))
        absolute_humidity_values = (humidity_values / 100) * saturation_vapor_pressure_values / (self.k_B * temperature_values_K)
        return absolute_humidity_values

    def calculate_water_vapor_molecules(self, absolute_humidity_values):
        r = 5.4745 #m
        h = 3160 #m
        number_of_water_molecules = (absolute_humidity_values * np.pi * r**2 * h) / self.M_water
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

# Function to plot the profile
def plot_profile(layers, values, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, 6))
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
            
    plt.savefig('atmospheric_density_profile.svg', format='svg', transparent=True)
    plt.legend()
    plt.show()

# Plot the profiles
'''plot_profile(layers, absolute_humidity_values, 'Absolute Humidity Profile', 'log Density (kg/m^3)')
plot_profile(layers, number_of_water_molecules, 'Number of Water Vapor Molecules Profile', 'Number of Molecules (molecules/m³)')
plot_profile(layers, total_atmospheric_density_values, 'Atmospheric Density Profile', 'Density (kg/m^3)')'''


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

#print(f"wind values: {wind_values, wind_slopes}")

# Function to plot each parameter
def plot_parameter(layers, values, slopes, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(values, layers, 'o', color = 'blue')
    for i in range(len(layers) - 1):
        ax.plot([values[i], values[i] + slopes[i] * (layers[i + 1] - layers[i])], [layers[i], layers[i + 1]], color = 'cornflowerblue')
    ax.set_yscale('log')
    ax.set_ylabel('log Altitude (km)')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True)
    plt.legend()

    # Add layer boundaries as dotted lines
    '''for layer in layers:
        ax.axhline(y=layer, color='cadetblue', linestyle='dotted')
    '''
    ax.axhline(y = layers[4], color = 'violet', linestyle='dotted', label = 'Top of Troposphere')
    ax.axhline(y = layers[0], color = 'limegreen', linestyle='dotted', label = 'Mauna Kea Summit')
    plt.legend()
    plt.show()

# Plot each parameter
'''plot_parameter(layers, temperature_values, temperature_slopes, 'Temperature Profile', 'Temperature (°C)')
plot_parameter(layers, humidity_values, humidity_slopes, 'Humidity Profile', 'Relative Humidity (%)')
plot_parameter(layers, emissivity_values, emissivity_slopes, 'Emissivity Profile', 'Atmospheric Emissivity')
plot_parameter(layers, pressure_values, pressure_slopes, 'Pressure Profile', 'Pressure (hPa)')
plot_parameter(layers, cn2_values, cn2_slopes, 'Cn^2 Profile', 'Cn^2 (m^-2/3)')
plot_parameter(layers, wind_values, wind_slopes, 'Wind Profile', 'Wind Speed (m/s)')'''
