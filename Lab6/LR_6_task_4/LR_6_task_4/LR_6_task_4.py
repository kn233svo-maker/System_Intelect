import csv
import ssl
import urllib.request
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
context = ssl._create_unverified_context()

rows = []
with urllib.request.urlopen(url, context=context) as response:
    lines = [line.decode('utf-8') for line in response.readlines()]
    reader = csv.DictReader(lines)
    for row in reader:
        # Відбір записів без пропущених значень
        if (row['price'] and row['price'] != 'NA' and
                row['train_class'] and row['train_class'] != 'NA' and
                row['fare'] and row['fare'] != 'NA' and
                row['origin'] and row['destination'] and row['train_type']):
            rows.append(row)

print(f"Завантажено коректних записів: {len(rows)}")

# Прямий байєсівський аналіз
total = len(rows)
turista_rows = [r for r in rows if r['train_class'] == 'Turista']
ave_rows = [r for r in rows if r['train_type'] == 'AVE']
turista_and_ave = [r for r in rows if r['train_class'] == 'Turista' and r['train_type'] == 'AVE']

p_turista = len(turista_rows) / total
p_ave = len(ave_rows) / total
p_ave_given_turista = len(turista_and_ave) / len(turista_rows)
p_turista_given_ave = (p_ave_given_turista * p_turista) / p_ave

print("\nБайєсівський аналіз ймовірностей:")
print(f"Апріорна ймовірність P(Turista) = {p_turista:.4f}")
print(f"Ймовірність поїзда AVE P(AVE) = {p_ave:.4f}")
print(f"Умовна ймовірність P(AVE | Turista) = {p_ave_given_turista:.4f}")
print(f"Апостеріорна ймовірність за теоремою Байєса P(Turista | AVE) = {p_turista_given_ave:.4f}")

# Підготовка ознак та навчання класифікатора GaussianNB для прогнозування тарифу fare
fare_counts = Counter(r['fare'] for r in rows)
valid_fares = {f for f, cnt in fare_counts.items() if cnt >= 20}
filtered_rows = [r for r in rows if r['fare'] in valid_fares]

cat_features = np.array([
    [r['origin'], r['destination'], r['train_type'], r['train_class']]
    for r in filtered_rows
])
prices = np.array([float(r['price']) for r in filtered_rows])
fares = np.array([r['fare'] for r in filtered_rows])

encoder = OrdinalEncoder()
X_cat_encoded = encoder.fit_transform(cat_features)
X = np.column_stack([X_cat_encoded, prices])

X_train, X_test, y_train, y_test = train_test_split(
    X, fares, test_size=0.25, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nТочність класифікації тарифів: {accuracy_score(y_test, y_pred):.4f}")
print("\nЗвіт класифікації:")
print(classification_report(y_test, y_pred, zero_division=0))

# Візуалізація розподілу цін для основних класів вагонів
plt.figure(figsize=(9, 5))
for t_class in ['Turista', 'Preferente', 'Turista Plus']:
    class_prices = [float(r['price']) for r in rows if r['train_class'] == t_class]
    if class_prices:
        plt.hist(class_prices, bins=30, alpha=0.55, label=f"{t_class} (n={len(class_prices)})")

plt.xlabel("Ціна квитка EUR")
plt.ylabel("Кількість квитків")
plt.title("Розподіл цін на квитки за класами обслуговування")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()