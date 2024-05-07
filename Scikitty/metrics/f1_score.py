from sklearn.metrics import f1_score

def puntuacion_de_f1(y_test, y_pred, average='weighted'):
    return f1_score(y_test, y_pred, average=average)
