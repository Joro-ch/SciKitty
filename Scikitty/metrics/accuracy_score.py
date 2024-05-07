from sklearn.metrics import accuracy_score

def puntuacion_de_exactitud(y_test, y_pred):
    return accuracy_score(y_test, y_pred)
