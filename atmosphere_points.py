import numpy as np
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y, z, temperature, humidity, emissivity, pressure, wind, cn2):
        self.x = x
        self.y = y
        self.z = z
        self.temperature = temperature
        self.humidity = humidity
        self.emissivity = emissivity
        self.pressure = pressure
        self.wind = wind
        self.cn2 = cn2

    def __repr__(self):
        return (f"Point(x={self.x}, y={self.y}, z={self.z}, "
                f"temperature={self.temperature}, humidity={self.humidity}, "
                f"emissivity={self.emissivity}, pressure={self.pressure}, "
                f"wind={self.wind}, cn2={self.cn2})")

# defining atmospheric parameters from "surface" = 4,200 m Mauna Kea height
def temperature_profile(z):
    T_0 = 5  # Surface temperature in °C
    L = 6.5  # Lapse rate in °C/km
    return T_0 - L * z

def humidity_profile(z):
    H_0 = 17  # Surface relative humidity in %
    a = 0.2  # Decay constant in 1/km
    return H_0 * np.exp(-a * z)

def emissivity_profile(z):
    epsilon_0 = 0.7  # Surface emissivity
    B = 0.1  # Rate of decrease
    return epsilon_0 * np.exp(-B * z)

def pressure_profile(z):
    P_0 = 1012.75  # Surface pressure in hPa
    M = 0.029  # Molar mass of Earth's air in kg/mol
    g = 9.81  # Gravity in m/s²
    R = 8.314  # Universal gas constant in J/(mol·K)
    T_m = 278.15  # Mean temperature in K
    return P_0 * np.exp(-M * g * z * 1000 / (R * T_m))

def wind_profile(z):
    W_0 = 16  # Base wind speed at ground level in m/s
    a = 8  # Scale factor
    b = 1.8  # Rate of increase with altitude
    return W_0 + a * np.log1p(b * z)

def cn2_profile(z):
    C_n2_0 = 1e-15  # Surface Cn^2 value in m^-2/3
    d = 0.2  # Decay constant in 1/km
    return C_n2_0 * np.exp(-d * z)

def generate_points(N=10000, decay_rate=2.0, max_altitude=10):
    points = []
    for _ in range(N):
        r = np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(0, max_altitude)

        z_weighted = np.random.exponential(scale=decay_rate)
        z_weighted = np.clip(z_weighted, 0, 10)

        temperature = temperature_profile(z_weighted)
        humidity = humidity_profile(z_weighted)
        emissivity = emissivity_profile(z_weighted)
        pressure = pressure_profile(z_weighted)
        wind = wind_profile(z_weighted)
        cn2 = cn2_profile(z_weighted)

        points.append(Point(x, y, z_weighted, temperature, humidity, emissivity, pressure, wind, cn2))
    return points

def plot_points(points):
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, marker='.', alpha=0.4)
    plt.show()

if __name__ == "__main__":
    points = generate_points()
    plot_points(points)
