from sklearn.metrics import recall_score

def puntuacion_de_recall(y_test, y_pred, average='weighted'):
    return recall_score(y_test, y_pred, average=average)
