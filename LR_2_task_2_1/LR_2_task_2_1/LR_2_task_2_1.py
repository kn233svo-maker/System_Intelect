import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

input_file = 'income_data.txt'

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 2500 

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        
        data = line[:-1].split(', ')
        
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
            
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X_features = X_encoded[:, :-1].astype(int)
y_target = X_encoded[:, -1].astype(int)

X_features = preprocessing.scale(X_features)

classifier = SVC(kernel='poly', degree=8)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=5)

classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

f1 = cross_val_score(classifier, X_features, y_target, scoring='f1_weighted', cv=3)
print("Поліноміальне ядро")
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")
print("Accuracy: " + str(round(100 * accuracy_score(y_test, y_test_pred), 2)) + "%")
print("Precision: " + str(round(100 * precision_score(y_test, y_test_pred, average='weighted', zero_division=0), 2)) + "%")
print("Recall: " + str(round(100 * recall_score(y_test, y_test_pred, average='weighted', zero_division=0), 2)) + "%")