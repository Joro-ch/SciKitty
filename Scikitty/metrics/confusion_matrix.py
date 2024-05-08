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
