import numpy as np
import pandas as pd

class Nodo:
    def __init__(self, es_hoja=False, regla=None, etiqueta=None, impureza=0, muestras=0):
        # Inicializa un nodo del árbol de decisión.
        self.es_hoja = es_hoja
        self.regla = regla
        self.etiqueta = etiqueta
        self.izquierda = None
        self.derecha = None
        self.impureza = impureza
        self.muestras = muestras

    def __str__(self):
        # Representación en string de un nodo.
        return f"Hoja: {self.etiqueta}" if self.es_hoja else f"Regla: {self.regla}"

class ArbolDecision:
    def __init__(self, caracteristicas, etiquetas, criterio='Entropía', min_muestras_div=2, max_profundidad=None):
        # Inicializa el árbol de decisión con los parámetros dados.
        self.caracteristicas = np.array(caracteristicas)
        self.etiquetas = np.array(etiquetas)
        self.nombres_caracteristicas = caracteristicas.columns.tolist() if isinstance(caracteristicas, pd.DataFrame) else [f'Característica[{i}]' for i in range(np.array(caracteristicas).shape[1])]
        self.criterio = criterio
        self.min_muestras_div = min_muestras_div
        self.max_profundidad = max_profundidad
        self.raiz = None

    def entrenar(self):
        # Entrena el árbol de decisión utilizando los datos proporcionados.
        self.raiz = self._construir_arbol(self.caracteristicas, self.etiquetas, 0)

    def _construir_arbol(self, caracteristicas, etiquetas, profundidad_actual):
        # Construye recursivamente el árbol de decisión.
        if self._deber_detener_division(etiquetas, caracteristicas.shape[0], profundidad_actual):
            return Nodo(es_hoja=True, etiqueta=self._etiqueta_mas_comun(etiquetas))

        mejor_regla, mejor_impureza = self._elegir_mejor_regla(caracteristicas, etiquetas)
        if not mejor_regla:
            return Nodo(es_hoja=True, etiqueta=self._etiqueta_mas_comun(etiquetas))

        indices_izquierda, indices_derecha = self._dividir(caracteristicas, mejor_regla)
        subarbol_izquierdo = self._construir_arbol(caracteristicas[indices_izquierda], etiquetas[indices_izquierda], profundidad_actual + 1)
        subarbol_derecho = self._construir_arbol(caracteristicas[indices_derecha], etiquetas[indices_derecha], profundidad_actual + 1)

        nodo = Nodo(regla=mejor_regla)
        nodo.izquierda = subarbol_izquierdo
        nodo.derecha = subarbol_derecho
        nodo.impureza = mejor_impureza
        nodo.muestras = caracteristicas.size
        return nodo

    def _deber_detener_division(self, etiquetas, num_muestras, profundidad_actual):
        # Determina si se debe detener la división del árbol en un nodo dado.
        if len(np.unique(etiquetas)) == 1 or num_muestras < self.min_muestras_div:
            return True
        if self.max_profundidad is not None and profundidad_actual >= self.max_profundidad:
            return True
        return False

    def _etiqueta_mas_comun(self, etiquetas):
        # Devuelve la etiqueta más común en un conjunto de etiquetas.
        valores, conteos = np.unique(etiquetas, return_counts=True)
        return valores[np.argmax(conteos)]

    def _elegir_mejor_regla(self, caracteristicas, etiquetas):
        # Selecciona la mejor regla de división para un conjunto de características y etiquetas.
        mejor_impureza = float('inf')
        mejor_regla = None
        n_muestras = len(etiquetas)
        lista_caracteristicas = caracteristicas.T

        for indice, caracteristica in enumerate(lista_caracteristicas):
            valores_unicos = np.unique(caracteristica)
            es_binaria = len(valores_unicos) == 2
            es_categorica = isinstance(caracteristica[0], str) and len(valores_unicos) > 2

            if not (es_binaria or es_categorica):
                continue

            for valor in valores_unicos:
                mascara_division = caracteristica == valor
                etiquetas_divididas = etiquetas[mascara_division]
                probabilidad_valor = len(etiquetas_divididas) / n_muestras
                impureza_valor = self._calcular_impureza(etiquetas_divididas)
                impureza = probabilidad_valor * impureza_valor

                mascara_no_division = caracteristica != valor
                etiquetas_no_divididas = etiquetas[mascara_no_division]
                probabilidad_no_valor = len(etiquetas_no_divididas) / n_muestras
                impureza_no_valor = self._calcular_impureza(etiquetas_no_divididas)
                impureza += probabilidad_no_valor * impureza_no_valor

                if impureza < mejor_impureza:
                    mejor_impureza = impureza
                    mejor_regla = (indice, '==', valor)

        return mejor_regla, mejor_impureza

    def _dividir(self, caracteristicas, regla):
        # Divide el conjunto de características según una regla dada.
        indice_columna, condicion, valor = regla
        indices_izquierda = np.where(caracteristicas[:, indice_columna] == valor)[0]
        indices_derecha = np.where(caracteristicas[:, indice_columna] != valor)[0]
        return indices_izquierda, indices_derecha

    def _calcular_impureza(self, etiquetas):
        # Calcula la impureza según el criterio especificado (entropía o índice de Gini).
        if self.criterio == 'Entropía':
            return self._calcular_entropia(etiquetas)
        else:
            return self._calcular_gini(etiquetas)

    def _calcular_entropia(self, etiquetas):
        # Calcula la entropía de un conjunto de etiquetas.
        _, conteos = np.unique(etiquetas, return_counts=True)
        probabilidades = conteos / conteos.sum()
        entropia = -np.sum(probabilidades * np.log2(probabilidades))
        return entropia

    def _calcular_gini(self, etiquetas):
        # Calcula el índice de Gini de un conjunto de etiquetas.
        _, conteos = np.unique(etiquetas, return_counts=True)
        probabilidades = conteos / conteos.sum()
        gini = 1 - np.sum(probabilidades ** 2)
        return gini

    def predecir(self, caracteristicas):
        # Predice las etiquetas para un conjunto de características dado.
        caracteristicas = np.array(caracteristicas)
        return [self._predecir_individual(caracteristica, self.raiz) for caracteristica in caracteristicas]

    def _predecir_individual(self, caracteristica, nodo):
        # Predice la etiqueta para una única instancia de características.
        if nodo.es_hoja:
            return nodo.etiqueta
        if self._seguir_regla(caracteristica, nodo.regla):
            return self._predecir_individual(caracteristica, nodo.izquierda)
        else:
            return self._predecir_individual(caracteristica, nodo.derecha)

    def _seguir_regla(self, caracteristica, regla):
        # Evalúa si una instancia de características cumple con una regla dada.
        indice_columna, condicion, valor = regla
        return caracteristica[indice_columna] == valor if condicion == '==' else caracteristica[indice_columna] != valor

    def imprimir_arbol(self, nodo=None, profundidad=0, condicion="Raíz"):
        # Imprime la estructura del árbol de decisión.
        if nodo is None:
            nodo = self.raiz

        if nodo.es_hoja:
            print(f"{'|   ' * profundidad}{condicion} -> Hoja: {nodo.etiqueta}")
        else:
            nombre_columna = self.nombres_caracteristicas[nodo.regla[0]]
            condicion_str = f"{nombre_columna} {nodo.regla[1]} {nodo.regla[2]}"
            print(f"{'|   ' * profundidad}{condicion} -> {condicion_str}")
            self.imprimir_arbol(nodo.izquierda, profundidad + 1, f"{condicion_str}")
            self.imprimir_arbol(nodo.derecha, profundidad + 1, f"No {condicion_str}")

    def obtener_estructura_arbol(self, nodo=None):
        # Devuelve la estructura del árbol de decisión en formato JSON.
        if nodo is None:
            nodo = self.raiz

        if nodo.es_hoja:
            return {"tipo": "Hoja", "etiqueta": nodo.etiqueta}
        else:
            nombre_columna = self.nombres_caracteristicas[nodo.regla[0]]
            info = f"""
                {nombre_columna} {nodo.regla[1]} {nodo.regla[2]}
                {self.criterio}: {nodo.impureza}
                muestras: {nodo.muestras}
                valor: [{nodo.izquierda.muestras}, {nodo.derecha.muestras}]
                clase: {nombre_columna}
            """

            return {
                "tipo": "Decision",
                "regla": info,
                "izquierda": self.obtener_estructura_arbol(nodo.izquierda),
                "derecha": self.obtener_estructura_arbol(nodo.derecha)
            }
