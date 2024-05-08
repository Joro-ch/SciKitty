from sklearn.metrics import f1_score

def puntuacion_de_f1(y_test, y_pred, average='weighted'):
    """
        Calcula la puntuación F1 del modelo de un árbol de decisión en base a "y_test", "y_pred" y "average".
        Usamos la implementación de SKLearn para este cálculo.
        En los scripts usamos solo weighted.

        Parámetros:
        y_test: Etiquetas verdaderas/Ground Truth.
        y_pred: Predicciones realizadas por el modelo. (Hacer predict antes)
        average: Tipo de promedio a usar para calcular la puntuación F1.
            - 'binary': Para problemas de clasificación binaria.
            - 'micro': Métrica global considerando el conteo total de verdaderos positivos, falsos negativos y falsos positivos.
            - 'macro': Promedio de la puntuación F1 de cada clase, sin considerar el desequilibrio de clases.
            - 'weighted': Promedio de la puntuación F1 de cada clase, ponderado por el número de muestras en cada clase.
            - 'samples': Promedio de la puntuación F1 de cada instancia.

        Retorna:
        float:
            Puntuación F1 del modelo según el tipo de promedio especificado.
    """
    return f1_score(y_test, y_pred, average=average)
