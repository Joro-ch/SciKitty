import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Divide un conjunto de datos en conjuntos de entrenamiento y prueba.

    Parámetros:
    - X: array-like, conjunto de características.
    - y: array-like, conjunto de etiquetas.
    - test_size: float, tamaño del conjunto de prueba (por defecto 0.2).
    - random_state: int, semilla aleatoria (por defecto None).

    Retorna:
    - X_train: array-like, conjunto de características de entrenamiento.
    - X_test: array-like, conjunto de características de prueba.
    - y_train: array-like, conjunto de etiquetas de entrenamiento.
    - y_test: array-like, conjunto de etiquetas de prueba.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Obtener el tamaño del conjunto de prueba
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Crear una permutación aleatoria de los índices
    indices = np.random.permutation(n_samples)
    
    # Dividir los índices en conjuntos de entrenamiento y prueba
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Dividir el conjunto de datos utilizando los índices
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test