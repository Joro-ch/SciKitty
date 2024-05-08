from sklearn.metrics import f1_score

def puntuacion_de_f1(y_test, y_pred, average='weighted'):
    """
        Se calcula el f1 del modelo de un árbol de decisión en
        base al "y_test", "y_pred" y el average.
    """
    return f1_score(y_test, y_pred, average=average)
