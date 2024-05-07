from sklearn.metrics import confusion_matrix

def matriz_de_confusion(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)
