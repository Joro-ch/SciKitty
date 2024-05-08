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
