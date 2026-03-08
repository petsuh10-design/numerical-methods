import csv
import numpy as np
import matplotlib.pyplot as plt
import os


# 1. Зчитування вхідних даних з перевіркою шляху
def read_data(filename):
    if not os.path.exists(filename):
        print(f"Помилка: Файл '{filename}' не знайдено у папці {os.getcwd()}")
        return np.array([]), np.array([])

    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))

    if len(x) == 0:
        print("Помилка: Файл порожній або неправильні заголовки (має бути n,t)")
    return np.array(x), np.array(y)


# 2. Знаходження розділених різниць [cite: 258, 263]
def get_divided_differences(x, y):
    n = len(y)
    if n == 0: return np.array([])
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef[0, :]


# 3. Многочлен Ньютона [cite: 306, 410]
def newton_poly(x_nodes, y_nodes, x_target):
    coef = get_divided_differences(x_nodes, y_nodes)
    if len(coef) == 0: return 0
    res = coef[0]
    product = 1.0
    for i in range(1, len(x_nodes)):
        product *= (x_target - x_nodes[i - 1])
        res += coef[i] * product
    return res


# --- Виконання ---
x_data, y_data = read_data("data.csv")

if len(x_data) > 0:
    target_tasks = 15000
    cost_newton = newton_poly(x_data, y_data, target_tasks)

    print(f"Прогноз вартості для {target_tasks} задач: ${cost_newton:.4f}")

    # Графік [cite: 412, 527]
    x_fine = np.linspace(min(x_data), max(x_data), 100)
    y_plot = [newton_poly(x_data, y_data, xi) for xi in x_fine]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='red', label='Дані (Варіант 4)')
    plt.plot(x_fine, y_plot, label='Інтерполяція Ньютона')
    plt.scatter([target_tasks], [cost_newton], color='green', label='Прогноз')
    plt.title("Прогноз вартості обчислень")
    plt.xlabel("Кількість завдань")
    plt.ylabel("Вартість ($)")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Програма зупинена через відсутність даних.")