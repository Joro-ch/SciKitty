import numpy as np
import pandas as pd

class Nodo:
    def __init__(self, es_hoja=False, regla=None, etiquetas=np.array([]), impureza=0):
        self.es_hoja = es_hoja
        self.regla = regla
        self.impureza = impureza
        self.etiquetas = etiquetas
        self.etiqueta = self._etiqueta_mas_comun(etiquetas)
        self.muestras = etiquetas.size
        self.izquierda = None
        self.derecha = None

    def __str__(self):
        return f"Hoja: {self.etiqueta}" if self.es_hoja else f"Regla: {self.regla}"
    
    def _etiqueta_mas_comun(self, etiquetas):

        if len(etiquetas) == 0:
            return None  # Si no hay etiquetas, retornar None o un valor indicativo

        valores, conteos = np.unique(etiquetas, return_counts=True)
        return valores[np.argmax(conteos)]

class DecisionStump:
    def __init__(self, caracteristicas, etiquetas, criterio='entropy', criterio_continuo='MSE'):
        self.caracteristicas = np.array(caracteristicas)
        self.etiquetas = np.array(etiquetas)
        
        if len(caracteristicas) > 0:
            self.nombres_caracteristicas = caracteristicas.columns.tolist() if isinstance(caracteristicas, pd.DataFrame) else [f'Característica[{i}]' for i in range(np.array(caracteristicas).shape[1])]
        else:
            self.nombres_caracteristicas = []

        self.criterio = criterio
        self.criterio_continuo = criterio_continuo
        self.raiz = None
        self.etiquetas_originales = np.unique(etiquetas)

        self.atributos_binarios_categoricos = []
        self.atributos_continuos = []

        self._clasificar_atributos()

    def _clasificar_atributos(self):
        for i, nombre in enumerate(self.nombres_caracteristicas):
            valores_unicos = np.unique(self.caracteristicas[:, i])
            if len(valores_unicos) <= 2 or isinstance(valores_unicos[0], str):
                self.atributos_binarios_categoricos.append(i)
            else:
                self.atributos_continuos.append(i)

    def fit(self):
        self.raiz = self._construir_stump(self.caracteristicas, self.etiquetas)

    def _construir_stump(self, caracteristicas, etiquetas):
        mejor_regla, mejor_impureza = self._elegir_mejor_regla(caracteristicas, etiquetas)
        
        nodo = Nodo(
            es_hoja=False,
            regla=mejor_regla,
            etiquetas=etiquetas,
            impureza=mejor_impureza
        )

        indices_izquierda, indices_derecha = self._dividir(caracteristicas, mejor_regla)
        
        nodo.izquierda = Nodo(
            es_hoja=True,
            etiquetas=etiquetas[indices_izquierda]
        )
        
        nodo.derecha = Nodo(
            es_hoja=True,
            etiquetas=etiquetas[indices_derecha]
        )

        return nodo

    def _elegir_mejor_regla(self, caracteristicas, etiquetas):
        mejor_impureza = float('inf')
        mejor_regla = None

        for indice in self.atributos_binarios_categoricos:
            caracteristica = caracteristicas[:, indice]
            valores_unicos = np.unique(caracteristica)
            for valor in valores_unicos:
                mascara_division = caracteristica == valor
                impureza = self._calcular_impureza_division(etiquetas, mascara_division)
                if impureza < mejor_impureza:
                    mejor_impureza = impureza
                    mejor_regla = (indice, '==', valor)

        for indice in self.atributos_continuos:
            caracteristica = caracteristicas[:, indice]
            valores_unicos = np.unique(caracteristica)
            valores_ordenados = np.sort(valores_unicos)
            puntos_medios = (valores_ordenados[:-1] + valores_ordenados[1:]) / 2
            for punto in puntos_medios:
                mascara_division = caracteristica <= punto
                impureza = self._calcular_impureza_division(etiquetas, mascara_division)
                if impureza < mejor_impureza:
                    mejor_impureza = impureza
                    mejor_regla = (indice, '<=', punto)

        return mejor_regla, mejor_impureza

    def _dividir(self, caracteristicas, regla):
        indice_columna, condicion, valor = regla
        if condicion == '<=':
            indices_izquierda = np.where(caracteristicas[:, indice_columna] <= valor)[0]
            indices_derecha = np.where(caracteristicas[:, indice_columna] > valor)[0]
        elif condicion == '==':
            indices_izquierda = np.where(caracteristicas[:, indice_columna] == valor)[0]
            indices_derecha = np.where(caracteristicas[:, indice_columna] != valor)[0]
        return indices_izquierda, indices_derecha

    def _calcular_impureza_division(self, etiquetas, mascara_division):
        impureza_valor, probabilidad_valor = self._calcular_impureza_y_probabilidad(etiquetas, mascara_division)
        impureza_no_valor, probabilidad_no_valor = self._calcular_impureza_y_probabilidad(etiquetas, ~mascara_division)
        impureza = probabilidad_valor * impureza_valor + probabilidad_no_valor * impureza_no_valor
        return impureza

    def _calcular_impureza_y_probabilidad(self, etiquetas, mascara):
        etiquetas_divididas = etiquetas[mascara]
        probabilidad = len(etiquetas_divididas) / len(etiquetas)
        impureza = self._calcular_impureza(etiquetas_divididas)
        return impureza, probabilidad

    def _calcular_impureza(self, etiquetas):
        valores_unicos = np.unique(etiquetas)

        if etiquetas.size == 0:
            return 0
        
        es_binaria = len(valores_unicos) <= 2
        es_categorica = isinstance(etiquetas[0], str) and len(valores_unicos) > 2

        if not (es_binaria or es_categorica):
            if isinstance(self.criterio_continuo, str):
                if self.criterio_continuo == 'MSE':
                    return self._calcular_mse(etiquetas)
            else:
                return self.criterio_continuo(etiquetas)
        elif isinstance(self.criterio, str):
            if self.criterio == 'entropy':
                return self._calcular_entropia(etiquetas)
            elif self.criterio == 'gini':
                return self._calcular_gini(etiquetas)
        else:
            return self.criterio(etiquetas)

    def _calcular_entropia(self, etiquetas):
        _, conteos = np.unique(etiquetas, return_counts=True)
        probabilidades = conteos / conteos.sum()
        return -np.sum(probabilidades * np.log2(probabilidades))

    def _calcular_gini(self, etiquetas):
        _, conteos = np.unique(etiquetas, return_counts=True)
        probabilidades = conteos / conteos.sum()
        return 1 - np.sum(probabilidades ** 2)

    def _calcular_mse(self, etiquetas):
        if etiquetas.size == 0:
            return 0
        
        media_etiquetas = np.mean(etiquetas)
        return np.mean((etiquetas - media_etiquetas) ** 2)

    def predict(self, caracteristicas):
        caracteristicas = np.array(caracteristicas)
        return [self._predict_individual(caracteristica, self.raiz) for caracteristica in caracteristicas]

    def _predict_individual(self, caracteristica, nodo):
        if nodo.es_hoja:
            return nodo.etiqueta

        if self._seguir_regla(caracteristica, nodo.regla):
            return nodo.izquierda.etiqueta
        else:
            return nodo.derecha.etiqueta

    def _seguir_regla(self, caracteristica, regla):
        indice_columna, condicion, valor = regla

        if condicion == '==':
            return caracteristica[indice_columna] == valor
        elif condicion == '<=':
            return caracteristica[indice_columna] <= valor
        else:
            return caracteristica[indice_columna] > valor
