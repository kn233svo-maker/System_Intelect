import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)

lin_reg_simple = LinearRegression()
lin_reg_simple.fit(X, y)
y_lin_pred = lin_reg_simple.predict(X)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print("X[0] =", X[0])
print("X_poly[0] =", X_poly[0])

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
y_poly_pred = lin_reg.predict(X_poly)

print("\nintercept_ =", lin_reg.intercept_)
print("coef_ =", lin_reg.coef_)

a = round(float(lin_reg.coef_[0][1]), 2)
b = round(float(lin_reg.coef_[0][0]), 2)
c = round(float(lin_reg.intercept_[0]), 2)
print(f"\nОтримана модель: y = {a}*x1^2 + {b}*x1 + {c}")

print("\nЯкість лінійної регресії:")
print("R2 =", round(r2_score(y, y_lin_pred), 3),
      "| MAE =", round(mean_absolute_error(y, y_lin_pred), 3),
      "| MSE =", round(mean_squared_error(y, y_lin_pred), 3))

print("Якість поліноміальної регресії:")
print("R2 =", round(r2_score(y, y_poly_pred), 3),
      "| MAE =", round(mean_absolute_error(y, y_poly_pred), 3),
      "| MSE =", round(mean_squared_error(y, y_poly_pred), 3))

X_new = np.linspace(-5, 1, 200).reshape(-1, 1)
X_new_poly = poly_features.transform(X_new)

plt.scatter(X, y, color='blue', s=18, label='Дані (Варіант 4)')
plt.plot(X_new, lin_reg_simple.predict(X_new), 'g--', linewidth=2, label='Лінійна регресія')
plt.plot(X_new, lin_reg.predict(X_new_poly), 'r-', linewidth=2.5, label='Поліноміальна регресія (deg=2)')
plt.xlabel('$x_1$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()