import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

# Визначення шляху до файлу вхідних даних та його завантаження
base_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_dir, 'data_clustering.txt')
X = np.loadtxt(input_file, delimiter=',')

# Оцінка ширини вікна bandwidth для вхідного набору даних X
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Створення та навчання моделі кластеризації методом зсуву середнього
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Витягування та виведення координат центрів кластерів
cluster_centers = meanshift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)

# Оцінка та виведення кількості знайдених кластерів
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data =", num_clusters)

# Відображення на графіку точок та центрів кластерів
plt.figure()
markers = 'o*xvs'
for i, marker in zip(range(num_clusters), cycle(markers)):
    # Відображення на графіку точок, що належать поточному кластеру
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color='black')

    # Відображення на графіку центру поточного кластера
    cluster_center = cluster_centers[i]
    plt.plot(
        cluster_center[0], cluster_center[1],
        marker='o', markerfacecolor='black',
        markeredgecolor='black', markersize=15
    )

plt.title('Кластери (Mean Shift)')
plt.show()