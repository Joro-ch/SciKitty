from sklearn.metrics import precision_score

def puntuacion_de_precision(y_test, y_pred, average='weighted'):
    return precision_score(y_test, y_pred, average=average)
