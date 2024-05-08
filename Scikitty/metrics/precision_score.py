from sklearn.metrics import precision_score

def puntuacion_de_precision(y_test, y_pred, average='weighted'):
    """
        Calcula la precisión del modelo de un árbol de decisión en base a "y_test", "y_pred" y el average.
        En los scripts usamos solo weighted.
        
        Parámetros:
        y_test: Etiquetas verdaderas/Ground Truth.
        y_pred: Predicciones realizadas por el modelo. (Hacer predict antes).
        average: string, [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
            Tipo de promedio a usar para calcular la precisión:
            - 'binary': Para problemas de clasificación binaria.
            - 'micro': Métrica global considerando el conteo total de verdaderos positivos, falsos negativos y falsos positivos.
            - 'macro': Promedio de la precisión de cada clase, sin considerar el desequilibrio de clases.
            - 'weighted': Promedio de la precisión de cada clase, ponderado por el número de muestras en cada clase.
            - 'samples': Promedio de la precisión de cada instancia.

        Retorna:
        float:
            Precisión del modelo según el tipo de promedio especificado.
    """
    return precision_score(y_test, y_pred, average=average)
