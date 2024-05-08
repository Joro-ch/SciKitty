import sklearn.model_selection

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
        Se obtiene los datos "X_train, X_test, y_train, y_test" de
        un modelo de un árbol de decisión.
    """
    return sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=random_state)
