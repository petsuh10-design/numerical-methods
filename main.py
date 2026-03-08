import requests
import numpy as np
import matplotlib.pyplot as plt

# 1. Запит до API
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
response = requests.get(url)
data = response.json()
results = data["results"]


# Формула Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# Табуляція вузлів
print(f"Кількість вузлів: {len(results)}")
print("\nТабуляція вузлів:")
print(" i |  Latitude | Longitude | Elevation (m)")
for i, p in enumerate(results):
    print(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}")

# Координати та відстані
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0.0]
for i in range(1, len(results)):
    d = haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    distances.append(distances[-1] + d)

print("\nТабуляція (відстань, висота):")
print(" i | Distance (m) | Elevation (m)")
for i in range(len(distances)):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")


# Функція для побудови сплайна (Метод прогонки)
def get_spline_func(x_nodes, y_nodes):
    n = len(x_nodes) - 1
    h = np.diff(x_nodes)

    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        B[i] = 3 * ((y_nodes[i + 1] - y_nodes[i]) / h[i] - (y_nodes[i] - y_nodes[i - 1]) / h[i - 1])

    c = np.linalg.solve(A, B)

    a = y_nodes[:-1]
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    def spline(val):
        idx = np.searchsorted(x_nodes, val) - 1
        idx = max(0, min(idx, n - 1))
        dx = val - x_nodes[idx]
        return a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3

    return spline


# Побудова графіків та розрахунок похибок
x_e = np.array(distances)
y_e = np.array(elevations)
f_etalon = get_spline_func(x_e, y_e)

plt.figure(1, figsize=(10, 5))
plt.title("Вплив кількості вузлів")
plt.plot(x_e, y_e, label="21 вузол (еталон)", linewidth=2)

plt.figure(2, figsize=(10, 5))
plt.title("Похибка апроксимації")

for count in [10, 15, 20]:
    idxs = np.linspace(0, len(x_e) - 1, count, dtype=int)
    xs, ys = x_e[idxs], y_e[idxs]

    f_s = get_spline_func(xs, ys)

    x_plot = np.linspace(x_e[0], x_e[-1], 300)
    y_plot = [f_s(tx) for tx in x_plot]

    y_at_etalon = np.array([f_s(tx) for tx in x_e])
    err = np.abs(y_e - y_at_etalon)

    print(f"\n{count} вузлів")
    print(f"Максимальна похибка: {np.max(err)}")
    print(f"Середня похибка: {np.mean(err)}")

    plt.figure(1)
    plt.plot(x_plot, y_plot, label=f"{count} вузлів")
    plt.figure(2)
    plt.plot(x_e, err, label=f"{count} вузлів")

plt.figure(1);
plt.legend();
plt.grid(True)
plt.figure(2);
plt.legend();
plt.grid(True)
plt.show()