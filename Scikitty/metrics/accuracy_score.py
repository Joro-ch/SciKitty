from sklearn.metrics import accuracy_score

def puntuacion_de_exactitud(y_test, y_pred):
    """
        Calcula la exactitud (accuracy) del modelo de un árbol de decisión en base a "y_test" y "y_pred".
        Usamos la implementación de SKLearn para este cálculo.
        
        Parámetros:
        y_test: Etiquetas verdaderas/Ground Truth.
        y_pred: Predicciones realizadas por el modelo. (Hacer predict antes)

        Retorna:
        float:
            Exactitud del modelo, que es la proporción de predicciones correctas sobre el total de predicciones.
    """
    return accuracy_score(y_test, y_pred)
