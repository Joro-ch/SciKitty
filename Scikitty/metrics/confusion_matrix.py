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
from sklearn.metrics import confusion_matrix

def matriz_de_confusion(y_test, y_pred):
    """
        Calcula la matriz de confusión del modelo de un árbol de decisión en base a "y_test" y "y_pred".
        Usamos la implementación de SKLearn para este cálculo.

        Parámetros:
        y_test: Etiquetas verdaderas/Ground Truth.
        y_pred: Predicciones realizadas por el modelo. (Hacer predict antes)

        Retorna:
        numpy.ndarray:
            Matriz de confusión donde las filas representan las clases verdaderas y las columnas las predicciones del modelo.
    """
    return confusion_matrix(y_test, y_pred)
