from sklearn.metrics import precision_score

def puntuacion_de_precision(y_test, y_pred, average='weighted'):
    """
        Se calcula el precision del modelo de un árbol de decisión en
        base al "y_test", "y_pred" y el average.
    """
    return precision_score(y_test, y_pred, average=average)
