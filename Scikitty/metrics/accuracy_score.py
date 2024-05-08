from sklearn.metrics import accuracy_score

def puntuacion_de_exactitud(y_test, y_pred):
    """
        Se calcula el accuracy del modelo de un árbol de decisión en
        base al "y_test" y al "y_pred".
    """
    return accuracy_score(y_test, y_pred)
