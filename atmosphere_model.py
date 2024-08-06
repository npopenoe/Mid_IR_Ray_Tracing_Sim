import numpy as np
import matplotlib.pyplot as plt

# Constants
k_B = 1.38e-23  # Boltzmann constant in J/K
R = 8.314  # Universal gas constant in J/(mol·K)
M_air = 0.029  # Molar mass of Earth's air in kg/mol
M_water = 0.018  # Molar mass of water vapor in kg/mol

# Define the layers and their altitudes for the troposphere up to 20 km
layers = np.linspace(4.2, 20, 6)  # 5 sections from 4.2 km to 20 km
midpoints = (layers[:-1] + layers[1:]) / 2  # Midpoints of the layers

# Define the parameter functions
def temperature_profile(z):
    T_0 = 15  # Surface temperature in °C
    L = 6.5  # Lapse rate in °C/km
    return T_0 - L * z/2

def humidity_profile(z):
    H_0 = 50  # Surface relative humidity in % at 4200 m
    a = 0.2  # Decay constant in 1/km
    return H_0 * np.exp(-a * (z - 4.2))

def emissivity_profile(z):
    epsilon_0 = 0.7  # Surface emissivity at 4200 m
    B = 0.1  # Rate of decrease
    return epsilon_0 * np.exp(-B * (z - 4.2))

def pressure_profile(z):
    P_0 = 1012.75  # Surface pressure in hPa at 4200 m
    P_0 = P_0 * 100  # Convert to Pa
    g = 9.81  # Gravity in m/s²
    T_m = temperature_profile(z) + 273.15  # Convert to Kelvin
    return P_0 * np.exp(-M_air * g * (z - 4.2) * 1000 / (R * T_m))

def wind_profile(z):
    W_0 = 16  # Base wind speed at 4200 m in m/s
    a = 5  # Adjusted scale factor for more realistic increase
    b = 1.5  # Adjusted rate of increase with altitude
    return W_0 + a * np.log1p(b * (z - 4.2))

def cn2_profile(z):
    C_n2_0 = 1e-15  # Surface Cn^2 value in m^-2/3 at 4200 m
    d = 0.2  # Decay constant in 1/km
    return C_n2_0 * np.exp(-d * (z - 4.2))

def saturation_vapor_pressure(T):
    return 0.61121 * np.exp((18.678 - T / 234.5) * (T / (257.14 + T)))

# Compute the values at the boundary points of the layers
temperature_values = temperature_profile(layers)  # In Celsius
temperature_values_K = temperature_values + 273.15
humidity_values = humidity_profile(layers)
emissivity_values = emissivity_profile(layers)
pressure_values = pressure_profile(layers)
wind_values = wind_profile(layers)
cn2_values = cn2_profile(layers)

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

# Function to calculate air density using ideal gas law
def air_density(P, T):
    return P * M_air / (R * T)  # Correct units with universal gas constant

# Compute air density
air_density_values = air_density(pressure_values, temperature_values_K)

# Compute absolute humidity (water vapor density)
saturation_vapor_pressure_values = saturation_vapor_pressure(temperature_values)
absolute_humidity_values = (humidity_values / 100) * saturation_vapor_pressure_values * M_water / (k_B * temperature_values_K)

print(saturation_vapor_pressure_values, absolute_humidity_values)

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

'''# Plot each parameter
plot_parameter(layers, temperature_values, temperature_slopes, 'Temperature Profile', 'Temperature (°C)')
plot_parameter(layers, humidity_values, humidity_slopes, 'Humidity Profile', 'Relative Humidity (%)')
plot_parameter(layers, emissivity_values, emissivity_slopes, 'Emissivity Profile', 'Atmospheric Emissivity')
plot_parameter(layers, pressure_values, pressure_slopes, 'Pressure Profile', 'Pressure (hPa)')
plot_parameter(layers, wind_values, wind_slopes, 'Wind Profile', 'Wind Speed (m/s)')
plot_parameter(layers, cn2_values, cn2_slopes, 'Cn^2 Profile', 'Cn^2 (m^-2/3)')'''

# Function to plot the profile
def plot_profile(layers, values, title, xlabel):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(values, layers, 'o', color='blue', linestyle ='-')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('log Altitude (km)')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True)

    # Add layer boundaries as dotted lines
    for i, layer in enumerate(layers):
        if i == 4:  # 5th from the bottom (index 4 in a 0-indexed array)
            ax.axhline(y=layer, color='pink', linestyle='dotted', label='Top of Troposphere')
            ax.axhline(y = layers[0], color = 'limegreen', linestyle='dotted', label = 'Mauna Kea Summit')
        else:
            ax.axhline(y=layer, color='paleturquoise', linestyle='dotted')
    
    plt.legend()
    plt.show()

# Plot the water vapor density profile
plot_profile(layers, absolute_humidity_values, 'Water Vapor Density Profile', 'Density (kg/m^3)')