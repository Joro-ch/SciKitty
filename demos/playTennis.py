# --------------------------------------------------------------------------------- #
"""
    Lo siguiente es solo necesario para que Python reconozca el directorio Scikitty
    el cual está en un nivel superior a este archivo. Esto se hace solo porque los
    scripts están almacenados dentro de una carpeta "demos" en un nivel más profundo
    que la carpeta Scikitty.
"""

import sys
import os

# Obtener el directorio actual del script.
directorio_actual = os.path.dirname(os.path.abspath(__file__))
# Obtener el directorio superior.
directorio_superior = os.path.abspath(os.path.join(directorio_actual, os.pardir))
# Se agrega el directorio superior a los paths que reconoce este archivo python.
sys.path.append(directorio_superior)

# --------------------------------------------------------------------------------- #

from Scikitty.models.DecisionTree import DecisionTree
from Scikitty.view.TreeVisualizer import TreeVisualizer
from Scikitty.metrics.accuracy_score import puntuacion_de_exactitud
from Scikitty.metrics.precision_score import puntuacion_de_precision
from Scikitty.metrics.recall_score import puntuacion_de_recall
from Scikitty.metrics.f1_score import puntuacion_de_f1
from Scikitty.metrics.confusion_matrix import matriz_de_confusion
from Scikitty.model_selection.train_test_split import train_test_split
import pandas as pd

# Se almacena el nombre del archivo donde se guarda el dataset
file_name = 'playTennis'

# Cargar los datos
data = pd.read_csv(f'../datasets/{file_name}.csv')

# Preparar los datos
features = data.drop('Play Tennis', axis=1)  # Asume que 'Play Tennis' es la columna objetivo
labels = data['Play Tennis']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Crear e instanciar el árbol de decisión
dt = DecisionTree(X_train, y_train, criterio='Entropy', min_muestras_div=2, max_profundidad=5)
dt.fit()

# Visualizar el árbol
tree_structure = dt.get_tree_structure()
visualizer = TreeVisualizer()
visualizer.graph_tree(tree_structure)
visualizer.get_graph(f'{file_name}_tree', ver=True)

# Imprimir resultados
y_pred = dt.predict(X_test)

accuracy = puntuacion_de_exactitud(y_test, y_pred)
precision = puntuacion_de_precision(y_test, y_pred, average='weighted')
recall = puntuacion_de_recall(y_test, y_pred, average='weighted')
f1 = puntuacion_de_f1(y_test, y_pred, average='weighted')
conf_matrix = matriz_de_confusion(y_test, y_pred)

print("Exactitud:", accuracy)
print("Precisión:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Matriz de confusión:") 
print(conf_matrix)
print("Predicted Labels:", y_pred)
print("Actual Labels:", y_test.tolist())