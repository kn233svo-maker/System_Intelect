import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)


def plot_learning_curves(model, X, y, title):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []
    for m_size in range(1, len(X_train)):
        model.fit(X_train[:m_size], y_train[:m_size])
        y_train_predict = model.predict(X_train[:m_size])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m_size]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальний набір (train)")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Перевірочний набір (val)")
    plt.xlabel("Розмір навчального набору")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.ylim([0, 3])
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

plt.figure(1)
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, "Криві навчання: лінійна регресія")

plt.figure(2)
polynomial_regression_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_10, X, y, "Криві навчання: поліноміальна регресія (degree=10)")

plt.figure(3)
polynomial_regression_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_2, X, y, "Криві навчання: поліноміальна регресія (degree=2)")

plt.show()