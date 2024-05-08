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
#-----------------------SCRIPT fictional_disease SCIKITTY----------------------------#

"""
    Este script demuestra el uso de un modelo de árbol de decisión para clasificar
    enfermedades ficticias utilizando el módulo Scikitty.
    
    Funcionalidades demostradas:
    - Cargar un dataset.
    - Preparar y dividir los datos en conjuntos de entrenamiento y prueba.
    - Entrenar un modelo de árbol de decisión.
    - Visualizar el árbol de decisión generado.
    - Evaluar el modelo utilizando varias métricas.
    - Guardar y cargar el modelo de árbol de decisión en/desde un archivo JSON.
    - Verificar la equivalencia funcional entre el árbol original y el árbol cargado desde JSON.
"""
# --------------------------------------------------------------------------------- #

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
from Scikitty.persist.TreePersistence import TreePersistence
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
dt = DecisionTree(X_train, y_train, criterio='gini', min_muestras_div=2, max_profundidad=5)
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

# Guardar el árbol en un archivo JSON
TreePersistence.save_tree(dt, f'{file_name}.json')

# Cargar el árbol desde el archivo JSON
loaded_dt = TreePersistence.load_tree(f'{file_name}.json')
print("Visualizando el árbol cargado desde el archivo JSON...")
loaded_tree_structure = loaded_dt.get_tree_structure()
loaded_visualizer = TreeVisualizer()
loaded_visualizer.graph_tree(loaded_tree_structure)
loaded_visualizer.get_graph(f'{file_name}_loaded_tree', ver=True)

# Imprimir resultados del árbol cargado
loaded_y_pred = loaded_dt.predict(X_test)

loaded_accuracy = puntuacion_de_exactitud(y_test, loaded_y_pred)
loaded_precision = puntuacion_de_precision(y_test, loaded_y_pred, average='weighted')
loaded_recall = puntuacion_de_recall(y_test, loaded_y_pred, average='weighted')
loaded_f1 = puntuacion_de_f1(y_test, loaded_y_pred, average='weighted')
loaded_conf_matrix = matriz_de_confusion(y_test, loaded_y_pred)

print("Exactitud (árbol cargado):", loaded_accuracy)
print("Precisión (árbol cargado):", loaded_precision)
print("Recall (árbol cargado):", loaded_recall)
print("F1-score (árbol cargado):", loaded_f1)
print("Matriz de confusión (árbol cargado):")
print(loaded_conf_matrix)
print("Predicted Labels (árbol cargado):", loaded_y_pred)
print("Actual Labels:", y_test.tolist())
