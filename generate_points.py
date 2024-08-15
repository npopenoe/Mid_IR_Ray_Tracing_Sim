import numpy as np
import matplotlib.pyplot as plt
from atmosphere_model import AtmosphericModel, layers, number_of_water_molecules

class AtmosphericPoint:
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
        return (f"AtmosphericPoint(x={self.x}, y={self.y}, z={self.z}, "
                f"temperature={self.temperature}, humidity={self.humidity}, "
                f"emissivity={self.emissivity}, pressure={self.pressure}, "
                f"wind={self.wind}, cn2={self.cn2})")

def generate_points(atm_model, layers, num_molecules_per_layer, primary_mirror_radius=5.4745, scaling_factor=1e38):
    points = []
    base_height = 17  # Start generating points 17 meters above the base which is above secondary mirror

    for i in range(len(layers) - 1):
        num_points = int(num_molecules_per_layer[i] / scaling_factor)
        print(f"Layer {i + 1}: Generating {num_points} points")
        for _ in range(num_points):
            r = primary_mirror_radius * np.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.random.uniform(base_height, layers[i + 1])  # Ensure z starts from base_height

            temperature = atm_model.temperature_profile(z)
            humidity = atm_model.humidity_profile(z)
            emissivity = atm_model.emissivity_profile(z)
            pressure = atm_model.pressure_profile(z)
            wind = atm_model.wind_profile(z)
            cn2 = atm_model.cn2_profile(z)

            points.append(AtmosphericPoint(x, y, z, temperature, humidity, emissivity, pressure, wind, cn2))
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
    atm_model = AtmosphericModel()
    points = generate_points(atm_model, layers, number_of_water_molecules)
    if points:
        plot_points(points)
    else:
        print("No points generated.")
