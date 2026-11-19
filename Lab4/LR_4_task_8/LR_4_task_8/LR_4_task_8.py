import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.1, 0.3, 0.4, 0.6, 0.7], dtype=float)
y = np.array([3.2, 3.0, 1.0, 1.8, 1.9], dtype=float)

X_matrix = np.vander(x, N=5, increasing=True)

A = np.linalg.solve(X_matrix, y)

def poly_func(val):
    return sum(A[i] * (val ** i) for i in range(len(A)))

y_02 = poly_func(0.2)
y_05 = poly_func(0.5)

print("Матриця Вандермонда X:\n", np.round(X_matrix, 4))
print("\nКоефіцієнти полінома (a0..a4):\n", np.round(A, 4))
print(f"\nПоліном 4-го степеня:\nP(x) = {A[0]:.4f} + ({A[1]:.4f})*x + ({A[2]:.4f})*x^2 + ({A[3]:.4f})*x^3 + ({A[4]:.4f})*x^4")
print(f"\nЗначення в точці x = 0.2: P(0.2) = {y_02:.4f}")
print(f"Значення в точці x = 0.5: P(0.5) = {y_05:.4f}")

x_smooth = np.linspace(min(x), max(x), 300)
y_smooth = poly_func(x_smooth)

plt.figure(figsize=(8, 5))
plt.plot(x_smooth, y_smooth, color='blue', linewidth=2, label='Інтерполяційний поліном 4-го степеня')
plt.scatter(x, y, color='red', s=55, zorder=4, label='Вузли інтерполяції (5 точок)')
plt.scatter([0.2, 0.5], [y_02, y_05], color='green', marker='D', s=65, zorder=5,
            label=f'Проміжні точки: P(0.2)={y_02:.2f}, P(0.5)={y_05:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Інтерполяція функції поліномом 4-го степеня')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()