import numpy as np
import matplotlib.pyplot as plt

X = np.array([6.5, 4.4, 3.8, 3.5, 3.1, 3.0], dtype=float)
Y = np.array([-5.0, -4.0, 0.7, 1.25, 3.0, 5.0], dtype=float)
n = len(X)

A = np.array([
    [n, np.sum(X)],
    [np.sum(X), np.sum(X ** 2)]
])
B = np.array([np.sum(Y), np.sum(X * Y)])

beta_0, beta_1 = np.linalg.solve(A, B)

Y_pred = beta_0 + beta_1 * X
S = np.sum((Y - Y_pred) ** 2)

print(f"Кількість точок n = {n}")
print(f"Параметр beta_0 (зсув) = {beta_0:.4f}")
print(f"Параметр beta_1 (нахил) = {beta_1:.4f}")
print(f"Рівняння апроксимуючої прямої: y = {beta_0:.4f} + ({beta_1:.4f}) * x")
print(f"Мінімальна сума квадратів похибок S = {S:.4f}")

x_line = np.linspace(min(X) - 0.3, max(X) + 0.3, 200)
y_line = beta_0 + beta_1 * x_line

plt.figure(figsize=(8, 5))
plt.scatter(X, Y, color='blue', s=45, zorder=3, label='Експериментальні точки (Варіант 14)')
plt.plot(x_line, y_line, color='red', linewidth=2, label=f'МНК: y = {beta_0:.2f} {beta_1:+.2f}x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Апроксимація методом найменших квадратів (Варіант 14)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()