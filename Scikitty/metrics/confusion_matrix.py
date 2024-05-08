from sklearn.metrics import confusion_matrix

def matriz_de_confusion(y_test, y_pred):
    """
        Se calcula la matriz de confusión del modelo de un árbol de 
        decisión en base al "y_test" y al "y_pred".
    """
    return confusion_matrix(y_test, y_pred)
