import sklearn.model_selection

def train_test_split(X, y, test_size=0.2, random_state=None):
    return sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=random_state)
