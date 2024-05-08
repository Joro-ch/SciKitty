# --------------------------------------------------------------------------------- #
"""
    Autores:
    1) Nombre: John Rojas Chinchilla
       ID: 118870938
       Correo: john.rojas.chinchilla@est.una.ac.cr
       Horario: 1pm

    2) Nombre: Abigail Salas
       ID: 402570890
       Correo: abigail.salas.ramirez@est.una.ac.cr
       Horario: 1pm

    3) Nombre: Axel Monge Ramirez
       ID: 118640655
       Correo: axel.monge.ramirez@est.una.ac.cr
       Horario: 1pm

    4) Nombre: Andrel Ramirez Solis
       ID: 118460426
       Correo: andrel.ramirez.solis@est.una.ac.cr
       Horario: 1pm
"""
# --------------------------------------------------------------------------------- #
"""
-----------------------SCRIPT CO2_car_emision SCI-KIT LEARN----------------------------

    Este script demuestra el uso de varias funcionalidades en el módulo scikit-learn:
    - Cargar un dataset.
    - Codificar variables categóricas.
    - Preparar y dividir los datos en conjuntos de entrenamiento y prueba.
    - Entrenar un modelo de árbol de decisión.
    - Visualizar el árbol de decisión.
    - Evaluar el modelo utilizando varias métricas.
"""
# --------------------------------------------------------------------------------- #

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
plt.figure(figsize=(11, 100))
plot_tree(dt, filled=True, feature_names=X_train.columns.tolist(), class_names=list(map(str, dt.classes_)))
plt.savefig(f'{file_name}_tree-scikitlearn.png')

# Imprimir resultados
y_pred = dt.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n------------------------------ ARBOL SCI-KIT ------------------------------\n")
print("Exactitud:", accuracy)
print("Precisión:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Matriz de confusión:") 
print(conf_matrix)
print("Etiquetas predichas:", y_pred)
print("Etiquetas reales:", y_test.tolist())
print("\nVisualizando el árbol de Sci-Kit Learn...\n")
plt.show()