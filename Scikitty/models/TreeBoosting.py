import numpy as np
import pandas as pd
from Scikitty.models.DecisionStump import DecisionStump

class TreeBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, criterio='entropy', criterio_continuo='MSE'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterio = criterio
        self.criterio_continuo = criterio_continuo
        self.stumps = []
        self.classes_ = None
        self.is_classification = False

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        if y.dtype == 'O':  # Para etiquetas categóricas
            self.is_classification = True
            y = np.where(y == self.classes_[1], 1, 0)  # Convertir a 0 y 1 para boosting
        
        residuals = y.copy()

        for _ in range(self.n_estimators):
            stump = DecisionStump(X, residuals, criterio=self.criterio, criterio_continuo=self.criterio_continuo)
            stump.fit()
            predictions = stump.predict(X)

            if self.is_classification:
                residuals = residuals - self.learning_rate * (2 * (predictions == 1) - 1)
            else:
                predictions = np.array(predictions).astype(float)
                residuals -= self.learning_rate * predictions  # Residuos de regresión

            self.stumps.append(stump)

    def predict(self, X):
        X = np.array(X)
        if self.is_classification:
            y_pred = np.zeros(X.shape[0])
        else:
            y_pred = np.zeros(X.shape[0], dtype=float)

        for stump in self.stumps:
            stump_predictions = np.array(stump.predict(X))
            if self.is_classification:
                y_pred += self.learning_rate * (2 * (stump_predictions == 1) - 1)
            else:
                y_pred += self.learning_rate * stump_predictions.astype(float)

        if self.is_classification:
            y_pred = np.where(y_pred > 0, self.classes_[1], self.classes_[0])

        return y_pred.astype(self.classes_.dtype)