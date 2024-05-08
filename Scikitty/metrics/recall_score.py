from sklearn.metrics import recall_score

def puntuacion_de_recall(y_test, y_pred, average='weighted'):
    """
        Se calcula el recall del modelo de un árbol de decisión en
        base al "y_test", "y_pred" y el average.
    """
    return recall_score(y_test, y_pred, average=average)
