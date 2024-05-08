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
from sklearn.metrics import recall_score

def puntuacion_de_recall(y_test, y_pred, average='weighted'):
    """
        Calcula el recall del modelo de un árbol de decisión en base a "y_test", "y_pred" y "average".
        Usamos la implementación de SKLearn para este cálculo.

        Parámetros:
        y_test: Etiquetas verdaderas/Ground Truth.
        y_pred: Predicciones realizadas por el modelo. (Hacer predict antes)
        average: Tipo de promedio a usar para calcular el recall.
            - 'binary': Para problemas de clasificación binaria.
            - 'micro': Métrica global considerando el conteo total de verdaderos positivos, falsos negativos y falsos positivos.
            - 'macro': Promedio del recall de cada clase, sin considerar el desequilibrio de clases.
            - 'weighted': Promedio del recall de cada clase, ponderado por el número de muestras en cada clase.
            - 'samples': Promedio del recall de cada instancia.

        Retorna:
        float:
            Recall del modelo según el tipo de promedio especificado.
    """
    return recall_score(y_test, y_pred, average=average)
