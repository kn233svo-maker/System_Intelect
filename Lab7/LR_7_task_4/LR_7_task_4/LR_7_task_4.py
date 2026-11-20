import os
import ssl
import json
import csv
import urllib.request
import numpy as np
from sklearn import covariance, cluster

base_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_dir, 'company_symbol_mapping.json')

# Завантаження прив'язок символів компаній до їх повних назв
with open(input_file, 'r', encoding='utf-8') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Завантаження архівних даних котирувань, заміна видаленого matplotlib.finance
url = "https://raw.githubusercontent.com/scikit-learn/examples-data/master/financial-data/{}.csv"
context = ssl._create_unverified_context()

opening_quotes, closing_quotes, valid_names = [], [], []
for symbol, name in zip(symbols, names):
    try:
        with urllib.request.urlopen(url.format(symbol), context=context) as response:
            reader = csv.DictReader(line.decode('utf-8') for line in response.readlines())
            rows = [{k.lower(): v for k, v in row.items()} for row in reader]
            opening_quotes.append([float(r['open']) for r in rows])
            closing_quotes.append([float(r['close']) for r in rows])
            valid_names.append(name)
    except Exception:
        continue

names = np.array(valid_names)
opening_quotes = np.array(opening_quotes, dtype=float)
closing_quotes = np.array(closing_quotes, dtype=float)

# Обчислення різниці між котируваннями при відкритті та закритті біржі
quotes_diff = closing_quotes - opening_quotes

# Нормалізація даних
X = quotes_diff.copy().T
X /= X.std(axis=0)

edge_model = covariance.GraphicalLassoCV()

# Навчання моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Створення моделі кластеризації на основі поширення подібності
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
num_labels = labels.max()

for i in range(num_labels + 1):
    print("Cluster", i + 1, "==>", ', '.join(names[labels == i]))