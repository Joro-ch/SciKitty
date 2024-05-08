from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Se almacena el nombre del archivo donde se guarda el dataset
file_name = 'CO2_car_emision'

# Cargar los datos
data = pd.read_csv(f'../datasets/{file_name}.csv')

# Codificar variables categóricas
data_encoded = pd.get_dummies(data, columns=['Car', 'Model'])

# Preparar los datos
features = data_encoded.drop('CO2', axis=1)
labels = data_encoded['CO2']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Crear e instanciar el árbol de decisión
dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=2, max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# Visualizar el árbol
plt.figure(figsize=(120, 80))
plot_tree(dt, filled=True, feature_names=X_train.columns.tolist(), class_names=list(map(str, dt.classes_)))
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
print("Etiquetas predichas:", y_pred)
print("Etiquetas reales:", y_test.tolist())

