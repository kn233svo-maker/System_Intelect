import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, silhouette_score

# Завантаження набору даних Iris
iris = load_iris()
X = iris['data']
y = iris['target']

# Створення екземпляра KMeans із повною демонстрацією основних гіперпараметрів
demo_kmeans = KMeans(
    n_clusters=8, init='k-means++', n_init=10, max_iter=300,
    tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd'
)

# Ініціалізація класифікатора KMeans для розбиття вибірки на 5 кластерів
kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)

kmeans.fit(X)

# Прогнозування індексів кластерів для кожного об'єкта вибірки X
y_kmeans = kmeans.predict(X)

plt.figure(figsize=(12, 9))

# Візуалізація точок даних для 5 кластерів та відображення координат їхніх центроїдів
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('KMeans (n_clusters=5)')
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')

# Визначення власної функції пошуку кластерів методом к-середніх
def find_clusters(X, n_clusters, rseed=2):
    # Ініціалізація генератора випадкових чисел та випадковий вибір початкових центрів кластерів
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # Призначення кожної точки даних до найближчого центру кластера за евклідовою відстанню
        labels = pairwise_distances_argmin(X, centers)

        # Перерахунок нових центроїдів як середнього арифметичного координат точок кожного кластера
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        # Перевірка умови збіжності
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

# Пошук 3 кластерів власною функцією та побудова діаграми розсіювання
centers, labels = find_clusters(X, 3)
plt.subplot(2, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('find_clusters (n_clusters=3, rseed=2)')
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')

# Пошук 3 кластерів власною функцією з іншим початковим зерном та візуалізація
centers, labels = find_clusters(X, 3, rseed=0)
plt.subplot(2, 2, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('find_clusters (n_clusters=3, rseed=0)')
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')

# Кластеризація на 3 кластери засобами бібліотеки scikit-learn та візуалізація
kmeans_3 = KMeans(n_clusters=3, n_init=10, random_state=0)
labels = kmeans_3.fit_predict(X)
plt.subplot(2, 2, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(kmeans_3.cluster_centers_[:, 0], kmeans_3.cluster_centers_[:, 1], c='black', s=200, alpha=0.5)
plt.title('KMeans sklearn (n_clusters=3, k-means++)')
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')

print(f"Коефіцієнт силуету (n_clusters=5): {silhouette_score(X, y_kmeans):.4f}")
print(f"Коефіцієнт силуету (n_clusters=3, k-means++): {silhouette_score(X, labels):.4f}")

plt.tight_layout()
plt.show()