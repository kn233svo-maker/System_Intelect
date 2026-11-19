import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Завантаження даних із цінами на нерухомість
try:
    housing_data = datasets.load_boston()
    X_raw, y_raw = housing_data.data, housing_data.target
    feature_names = np.array(housing_data.feature_names)
except (AttributeError, ImportError):
    housing_data = datasets.fetch_openml(data_id=531, as_frame=False, parser='auto')
    X_raw = np.asarray(housing_data.data, dtype=float)
    y_raw = np.asarray(housing_data.target, dtype=float)
    feature_names = np.array(housing_data.feature_names)

# Перемішування даних для об'єктивності аналізу
X, y = shuffle(X_raw, y_raw, random_state=7)

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)

# Створення та навчання моделі на основі регресора AdaBoost
regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    n_estimators=400,
    random_state=7
)
regressor.fit(X_train, y_train)

# Обчислення показників ефективності регресора AdaBoost
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Вилучення та нормалізація значень важливості ознак
feature_importances = regressor.feature_importances_
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Сортування ознак за спаданням відносної важливості
index_sorted = np.flipud(np.argsort(feature_importances))

print("\nВідносна важливість ознак (%):")
for idx in index_sorted:
    print(f"{feature_names[idx]:<10}: {feature_importances[idx]:.2f}%")

# Розміщення міток уздовж осі X та побудова стовпчастої діаграми
pos = np.arange(index_sorted.shape[0]) + 0.5

plt.figure(figsize=(10, 5))
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted])
plt.ylabel('Relative Importance')
plt.title('Оцінка важливості ознак з використанням регресора AdaBoost')
plt.tight_layout()
plt.show()