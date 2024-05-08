from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt

# Se almacena el nombre del archivo donde se guarda el dataset
file_name = 'fictional_reading_place'

# Cargar los datos
data = pd.read_csv(f'datasets/{file_name}.csv')

# Codificar características categóricas
encoder = OneHotEncoder()
features_encoded = encoder.fit_transform(data.drop('user_action', axis=1)).toarray()
labels = data['user_action']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(features_encoded, labels, test_size=0.2, random_state=42)

# Crear e instanciar el árbol de decisión
dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=2, max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# Visualizar el árbol
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=encoder.get_feature_names_out().tolist(), class_names=dt.classes_.tolist(), filled=True)
plt.savefig(f'{file_name}_tree-scikitlearn.png')

# Imprimir resultados
y_pred = dt.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Exactitud:", accuracy)
print("Precisión:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Matriz de confusión:") 
print(conf_matrix)
print("Predicted Labels:", y_pred)
print("Actual Labels:", y_test.tolist())
