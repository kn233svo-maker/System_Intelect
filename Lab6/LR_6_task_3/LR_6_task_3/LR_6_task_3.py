import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB

X_raw = np.array([
    ['Sunny', 'High', 'Weak'],     
    ['Sunny', 'High', 'Strong'],     
    ['Overcast', 'High', 'Weak'],     
    ['Rain', 'High', 'Weak'],        
    ['Rain', 'Normal', 'Weak'],      
    ['Rain', 'Normal', 'Strong'],     
    ['Overcast', 'Normal', 'Strong'], 
    ['Sunny', 'High', 'Weak'],     
    ['Sunny', 'Normal', 'Weak'],   
    ['Rain', 'Normal', 'Weak'],      
    ['Sunny', 'Normal', 'Strong'],    
    ['Overcast', 'High', 'Strong'],   
    ['Overcast', 'Normal', 'Weak'],   
    ['Rain', 'High', 'Strong']       
])

# Цільова змінна Play
y_raw = np.array([
    'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
    'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'
])

# Кодування категоріальних ознак у числові індекси
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X_raw)

# Навчання наївного байєсівського класифікатора
model = CategoricalNB()
model.fit(X_encoded, y_raw)

sample = np.array([['Sunny', 'Normal', 'Strong']])
sample_encoded = encoder.transform(sample)

prediction = model.predict(sample_encoded)[0]
probabilities = model.predict_proba(sample_encoded)[0]

print("Умови прогнозу:", sample[0])
for cls, prob in zip(model.classes_, probabilities):
    print(f"Ймовірність '{cls}': {prob:.4f} ({prob * 100:.2f}%)")
print("Результат прогнозу:", prediction)