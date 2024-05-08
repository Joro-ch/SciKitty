import numpy as np
import pandas as pd

class Nodo:

    # Función de construción de la clase Nodo.
    def __init__(self, es_hoja=False, regla=None, etiqueta=None, impureza=0, etiquetas=np.array([])):
        """
            Inicializa un nodo del árbol de decisión, cada nodo tiene su regla de división además
            de un atributo que indica si es un nodo final (hoja). Nuestro algoritmo está diseñado
            para que las ramas del nodo siempre sean binarias, por lo que si hay un atributo multiclase
            los hijos del nodo corresponderían a si dicho atributo presenta la subclase o no.
            Etiqueta es el nombre del target, se utilizaa solo cuando el nodo es hoja al momento de
            representarlo gráficamente.
        """

        # Atributo del nodo para saber si se trata de un nodo hoja o no.
        self.es_hoja = es_hoja

        # Atributo que almacena la regla que el nodo realiza para hacer el split.
        self.regla = regla

        # Atributo que almacena la etiqueta más común del split.
        self.etiqueta = etiqueta

        # Atributo que almacena el subárbol izquierdo del nodo.
        self.izquierda = None

        # Atributo que almacena el subárbol derecho del nodo.
        self.derecha = None

        # Atributo que almacena la impureza del split del nodo actual.
        self.impureza = impureza

        # Atributo que alamacena las etiquetas que componen el split.
        self.etiquetas = etiquetas

        # Atributo que representa la cantidad de muestras que hay total en el split actual.
        self.muestras = etiquetas.size


    # Función de impresión de la clase nodo.
    def __str__(self):
        """
            Describe a un nodo dependiendo si es hoja o no. En caso de no ser hoja, se muestra la regla
            y en caso de ser hoja, se muestra la etiqueta (nombre del target).
        """
        return f"Hoja: {self.etiqueta}" if self.es_hoja else f"Regla: {self.regla}"


class DecisionTree:

    # Función de construcción de la clase DecisionTree.
    def __init__(self, caracteristicas, etiquetas, criterio='entropy', min_muestras_div=2, max_profundidad=None):
        """
            Definición del algoritmo de aprendizaje automático "Árbol de Decisión". La idea es construir un árbol donde 
            los nodos son preguntas o reglas sobre alguna característica del conjunto de datos (DS), dichas reglas,
            dividirán al DS en subconjuntos más pequeños según las preguntas o reglas que mejor dividan al DS.

            Nuestro árbol funciona escogiendo las características que generen los subconjuntos con menor impureza respecto a
            las etiquetas a predict, utilizando criterios como "gini" o "entropy" si los datos son multiclase o binarios
            y MSE si los datos son contínuos y requieren de técnicas de regresión.

            El árbol recibirá una lista de características que dividirá en los nombres de dichas características y sus
            valores, además de las etiquetas correctas a predict en estructuras de numpy.

            El árbol recibe como parámetro escrito si el criterio de impureza a utilizar es "gini" o "entropy" además de 
            hiperparámetros de regularización que el usuario definirá para controlar el modelo, como el máximo de niveles
            de profundidad del árbol y el mínimo de muestras que debe haber para hacer una nueva división de características.

            Se inicializa un nodo raíz que, en el proceso de fit al modelo, definirá su regla de división y nodos hijo.
        """

        # Atributo Numpy array con las caracteristicas o x_train del árbol.
        self.caracteristicas = np.array(caracteristicas)

        # Atributo Numpy array con las etiquetas o y_train del árbol.
        self.etiquetas = np.array(etiquetas)

        # Se comprueba si el numpy array de carcteristicas tiene un tamaño superior a 0.
        if len(caracteristicas) > 0:
            # Se asignan los nombres de las caracteristicas.
            self.nombres_caracteristicas = caracteristicas.columns.tolist() if isinstance(caracteristicas,
            pd.DataFrame) else [f'Característica[{i}]' for i in range(np.array(caracteristicas).shape[1])]
            # Si no se cumple, se dejan los nombre de las caracteristicas como un array vacío.
        else: self.nombres_caracteristicas = []

        # Atributo que almacena el criterio a utilizar para calcular la impureza de los splits.
        self.criterio = criterio
        
        # Atributo que almacena el hiperparametro de mínimo de muestras por split.
        self.min_muestras_div = min_muestras_div

        # Atributo que almacena el hiperparametro del máximo de profundidad del árbol.
        self.max_profundidad = max_profundidad

        # Atributo que almacena la raiz del árbol.
        self.raiz = None

        # Atributo que almacena las etiquetas originales o las etiquetas unicas del y_train.
        self.etiquetas_originales = np.unique(etiquetas)


    @classmethod
    def generar_arbol (cls, raiz):
        arbol= cls([],[])
        arbol.raiz=raiz
        return arbol
    

    def is_balanced(self, umbral=0.5):
        """
            Evalúa si el dataset está balanceado basándose en un umbral de balance.
            Un dataset se considera balanceado si la proporción de la clase minoritaria respecto
            a la clase mayoritaria, del target o etiquetas a predecir, es mayor o igual al umbral
            establecido por el usuario. Devuelve true o false según el dataset esté balanceado o
            no.
        """

        # Se obtiene la cantidad total de cada elemento diferente del y_train.
        _, conteos = np.unique(self.etiquetas, return_counts=True)

        # Si solo hay un solo tipo de elemento, el dataset no está balanceado.
        if len(conteos) == 1:
            return False
        
        # Se calcula la proporción entre el mínimo y el máximo.
        proporción_min_max = conteos.min() / conteos.max()

        # Se retorna un True si la proporción es mayor o igual al umbral planteado o False en caso contrario.
        return proporción_min_max >= umbral
    

    def fit(self):
        """
            Entrena el árbol de decisión utilizando los datos proporcionados. Llama al proceso de construir un árbol
            con las características y etiquetas.
        """

        # Se realizá el llamado a la función encargada de construir el árbol.
        self.raiz = self._construir_arbol(self.caracteristicas, self.etiquetas, 0)


    def _construir_arbol(self, caracteristicas, etiquetas, profundidad_actual):
        """
            Valida si se debe seguir dividiendo el conjunto de datos, en caso afirmativo, busca la mejor regla de
            división y divide el conjunto de datos en izquierda y derecha según la regla de división y llama 
            recursivamente a si mismo para construir el árbol de los nodos izquierda y derecha, teniendo cada uno
            de ellos un nuevo subconjunto de datos. En caso negativo, define el nodo como hoja y representará a una 
            etiqueta (la etiqueta más común que posea).
        """

        # Se obtiene la etiqueta que es más común para el split actual.
        nodo_etiqueta = self._etiqueta_mas_comun(etiquetas)

        # Se obtiene la impureza del split actual.
        nodo_impureza = self._calcular_impureza(etiquetas)

        # Caso base para la recursividad.
        # Valida los hiperparámetros: profundidad y la cantidad de la muestra.
        if self._detener_division(etiquetas, caracteristicas.shape[0], profundidad_actual):
            return Nodo(es_hoja=True, etiqueta=nodo_etiqueta, impureza=nodo_impureza, etiquetas=etiquetas)
        
        # Se obtiene la característica con la mejor pureza.
        mejor_regla, _ = self._elegir_mejor_regla(caracteristicas, etiquetas)

        # Se comprueba si no exite la mejor regla para poder crear el nodo si es el caso.
        if not mejor_regla:
            return Nodo(es_hoja=True, etiqueta=nodo_etiqueta, impureza=nodo_impureza, etiquetas=etiquetas)
        
        # Se inicializan dos variables que almacenaran los indicies de cada lado del split.
        indices_izquierda = indices_derecha = 0

        # Se comprueba si existe la mejor regla para obtener los indices de cada lado del split.
        if mejor_regla:
            # Se generan las divisiones recursivamente
            indices_izquierda, indices_derecha = self._dividir(caracteristicas, mejor_regla)
        
        # Se genera el subárbol izquierdo del nodo actual.
        subarbol_izquierdo = self._construir_arbol(
            caracteristicas[indices_izquierda], etiquetas[indices_izquierda], profundidad_actual + 1)
        
        # Se genera el subárbol derecho del nodo actual.
        subarbol_derecho = self._construir_arbol(
            caracteristicas[indices_derecha], etiquetas[indices_derecha], profundidad_actual + 1)
        
        # Al nodo raíz se le asigna la mejor característica según su impureza.
        nodo = Nodo(etiqueta=nodo_etiqueta, regla=mejor_regla, impureza=nodo_impureza, etiquetas=etiquetas)

        # Se le agrega el subárbol izquierdo.
        nodo.izquierda = subarbol_izquierdo

        # Se le agrega el subárbol derecho.
        nodo.derecha = subarbol_derecho
        
        return nodo


    def _detener_division(self, etiquetas, num_muestras, profundidad_actual):
        """
            Indica si hay alguna razón para detener el split, ya sea debido a hiperparámetros o debido a que el
            conjunto ya es totalmente puro.
        """

        # Si sólo hay una etiqueta o que el número de muestras sean menores al hiperparámetro.
        if len(np.unique(etiquetas)) == 1 or num_muestras < self.min_muestras_div:
            return True
        
        # Verifica que la profundidad actual sea mayor o igual a la máxima profundidad.
        if self.max_profundidad is not None and profundidad_actual >= self.max_profundidad:
            return True
        
        return False


    def _etiqueta_mas_comun(self, etiquetas):
        """
            Devuelve la etiqueta más común en un conjunto de etiquetas.
        """

        # Se crea un array que contiene la cantidad de conteos de cada etiqueta.
        valores, conteos = np.unique(etiquetas, return_counts=True)

        # Se obtiene el valor máximo del array.
        return valores[np.argmax(conteos)]


    def _elegir_mejor_regla(self, caracteristicas, etiquetas):
        """
            Encuentra la regla que genera la menor impureza respecto a las etiquetas a predict, usa el criterio
            escogido por el usuario (de momento el criterio es solo "gini" o "entropy") De momento no toma en cuenta
            el feature si este es continuo.

            Se evalúa cada valor único de la característica y se busca el que produzca la menor impureza respecto a 
            las etiquetas, tomando en cuenta si la característica presenta ese valor o no. Se utiliza un promedio ponderado
            con el cálculo de la impureza, eso consiste en calcular las probabilidades de que la característica presente ese valor único y 
            de que no lo presente, y multiplicar por la impureza que utiliza como probabilidades las de cada etiqueta respecto
            a si se presentan cuando el valor del atributo es el evaluado o no, respectivamente.
        """

        # Selecciona la mejor regla de división para un conjunto de características y etiquetas.
        mejor_impureza = float('inf')
        mejor_regla = None
        n_muestras = len(etiquetas)
        lista_caracteristicas = caracteristicas.T

        for indice, caracteristica in enumerate(lista_caracteristicas):

            # Se guardan la cantidad de cada caracterítica.
            valores_unicos = np.unique(caracteristica)

            # Verifica que la característica sea binaria.
            es_binaria = len(valores_unicos) <= 2

            # Se guarda el resultado de la característica en un str.
            # Si su longitud es mayor a 2, entonces es una palabra (variable categórica).
            es_categorica = isinstance(
                caracteristica[0], str) and len(valores_unicos) > 2
            
            if not (es_binaria or es_categorica):
                # Ordena los valores únicos y calcula los puntos medios posibles para realizar splits.
                valores_ordenados = np.sort(valores_unicos)
                puntos_medios = (valores_ordenados[:-1] + valores_ordenados[1:]) / 2
                
                for punto in puntos_medios:
                    mascara_division = caracteristica <= punto
                    etiquetas_divididas = etiquetas[mascara_division]
                    probabilidad_valor = len(etiquetas_divididas) / n_muestras
                    impureza_valor = self._calcular_impureza(etiquetas_divididas)
                    impureza = probabilidad_valor * impureza_valor

                    mascara_no_division = caracteristica > punto
                    etiquetas_no_divididas = etiquetas[mascara_no_division]
                    probabilidad_no_valor = len(etiquetas_no_divididas) / n_muestras
                    impureza_no_valor = self._calcular_impureza(etiquetas_no_divididas)
                    impureza += probabilidad_no_valor * impureza_no_valor

                    if impureza <= mejor_impureza:
                        mejor_impureza = impureza
                        mejor_regla = (indice, '<=', punto)
            else:
                # Si la característica no es binaria, se evalúa cada valor único de la característica.
                for valor in valores_unicos:
                    # Concepto que el profe vio ayer en clase.
                    # Se crea una máscara booleana para las características con el mismo valor.
                    mascara_division = caracteristica == valor
                        
                    etiquetas_divididas = etiquetas[mascara_division]
                    probabilidad_valor = len(etiquetas_divididas) / n_muestras
                    impureza_valor = self._calcular_impureza(
                        etiquetas_divididas)
                    impureza = probabilidad_valor * impureza_valor
                    # Crea una máscara booleana para las caracteríticas con diferente valor.
                    mascara_no_division = caracteristica != valor
                    etiquetas_no_divididas = etiquetas[mascara_no_division]
                    probabilidad_no_valor = len(
                        etiquetas_no_divididas) / n_muestras
                    impureza_no_valor = self._calcular_impureza(
                        etiquetas_no_divididas)
                    impureza += probabilidad_no_valor * impureza_no_valor
                    # Se busca la mejor impuera comparando la anterior con la actual.
                    if impureza <= mejor_impureza:
                        mejor_impureza = impureza
                        mejor_regla = (indice, '==', valor)
        return mejor_regla, mejor_impureza


    def _dividir(self, caracteristicas, regla):
        """
            Divide el conjunto de datos dependiendo si cumplen la regla o no.
        """

        # Se obtiene el indice de la columna, la condición y el valor a comparar de la regla.
        indice_columna, condicion, valor = regla

        # Encuentra los índices que cumplen la regla (izquierda) y las que no (derecha).
        if(condicion == '<='):
            indices_izquierda = np.where(
                caracteristicas[:, indice_columna] <= valor)[0]
            indices_derecha = np.where(
                caracteristicas[:, indice_columna] > valor)[0]
        elif(condicion == '=='):
            indices_izquierda = np.where(
                caracteristicas[:, indice_columna] == valor)[0]
            indices_derecha = np.where(
                caracteristicas[:, indice_columna] != valor)[0]
        return indices_izquierda, indices_derecha


    def _calcular_impureza(self, etiquetas):
        """
            Escoge que criterio usar y devuelve la impureza calculada respecto a las etiquetas
            dependiendo del criterio escogido por el usuario en la definición del árbol de decisión
            para etiquetas multiclase o binarias, o MSE para etiquetas contínuas (target contínuo).
        """

        # Se optienen los valores unicos de las etiquetas.
        valores_unicos = np.unique(etiquetas)

        # Se comprueba si la cantidad de etiquetas es igual a cero para retornar 0 dado el caso.
        if etiquetas.size == 0:
            return 0
        
        # Se determina si las etiquetas son continuas (no categoricas y no binarias).
        es_binaria = len(valores_unicos) <= 2
        es_categorica = isinstance(etiquetas[0], str) and len(valores_unicos) > 2

        # Dependiento del "es_binaria" o "es_categorica" y del críterio del árbol
        # se calcula la impureza con un críterio diferente.
        if not (es_binaria or es_categorica):
            return self._calcular_mse(etiquetas)
        elif self.criterio == 'entropy':
            return self._calcular_entropia(etiquetas)
        elif self.criterio == 'gini':
            return self._calcular_gini(etiquetas)

    def _calcular_entropia(self, etiquetas):
        """
            Devuelve la impureza utilizando las probabilidades de cada etiqueta usando el criterio entropía.
        """

        # Se obtienen las cantidades totales de los valores unicos de las etiquetas.
        _, conteos = np.unique(etiquetas, return_counts=True)

        # Se calculan las probabilidades de cada etiqueta.
        probabilidades = conteos / conteos.sum()

        # Se calcula la entropia de las etiquetas.
        return -np.sum(probabilidades * np.log2(probabilidades))

    def _calcular_gini(self, etiquetas):
        """
            Devuelve la impureza utilizando las probabilidades de cada etiqueta usando el criterio gini.
        """

        # Se obtienen las cantidades totales de los valores unicos de las etiquetas.
        _, conteos = np.unique(etiquetas, return_counts=True)

        # Se calculan las probabilidades de cada etiqueta.
        probabilidades = conteos / conteos.sum()

        # Se calcula el indice gini de las etiquetas.
        return 1 - np.sum(probabilidades ** 2)


    def _calcular_mse(self, etiquetas):
        """
            Devuelve la impureza utilizando las probabilidades de cada etiqueta usando MSE, donde y_hat es el promedio de y.
        """

        # Se comprueba que la cantidad de etiquetas sea igual a cero para retornar 0 dado el caso.
        if etiquetas.size == 0:
            return 0
        
        # Se calcula la media de las etiquetas.
        media_etiquetas = np.mean(etiquetas)

        # Se calcula el mse de las etiquetas.
        return np.mean((etiquetas - media_etiquetas) ** 2)


    def predict(self, caracteristicas):
        """
            Devuelve las predicciones de cada instancia del Dataset.
        """

        # Se crea un numpu array utilizando las caracteristicas.
        caracteristicas = np.array(caracteristicas)

        # Se calculan las predicciones del modelos en base al x_train.
        return [self._predict_individual(caracteristica, self.raiz) for caracteristica in caracteristicas]


    def _predict_individual(self, caracteristica, nodo):
        """
            Determina la predicción para una instancia del dataset dependiendo si sus características cumplen
            las reglas de los nodos del árbol.
        """

        # Si es una hoja devuelve la etiqueta como predicción
        if nodo.es_hoja: 
            return nodo.etiqueta
        # Si se cumple con la regla, recursivamente valide por al subárbol izquierdo
        if self._seguir_regla(caracteristica, nodo.regla):
            return self._predict_individual(caracteristica, nodo.izquierda)
        # Si no se cumple la regla, recursivamente valide por el subárbol derecho
        else:
            return self._predict_individual(caracteristica, nodo.derecha)


    def _seguir_regla(self, caracteristica, regla):
        """
            Devuelve el booleano que indica si cumple o no la regla dependiendo si la regla es <= o ==.
        """

        # Se obtiene el indice de la columna, la condición y el valor del elemento a comparar en base a la regla.
        indice_columna, condicion, valor = regla

        # Comprueba si la característica cumple con la regla a seguir.
        if condicion == '==':
            return caracteristica[indice_columna] == valor
        elif condicion == '<=':
            return caracteristica[indice_columna] <= valor
        else:
            return caracteristica[indice_columna] > valor


    def imprimir_arbol(self, nodo=None, profundidad=0, condicion="Raíz"):
        """
            Imprime el árbol mediante prints.
        """

        # Se comprueba si el nodo es igual a None para igualar el nodo a la raiz.
        if nodo is None:
            nodo = self.raiz

        # Se comprueba si el nodo es una hoja para imprimir el fortmato de la hoja
        # en caso contrario se imprime en el formato del nodo.
        if nodo.es_hoja:
            print(f"{'|   ' * profundidad}{condicion} -> Hoja: {nodo.etiqueta}")
        else:
            # Toma el nombre de la característica según la regla actual
            nombre_columna = self.nombres_caracteristicas[nodo.regla[0]]
            condicion_str = f"{nombre_columna} {nodo.regla[1]} {nodo.regla[2]}"
            print(f"{'|   ' * profundidad}{condicion} -> {condicion_str}")
            # Llama recursivamente a la función con respectoal subárbol izquierdo y derecho
            self.imprimir_arbol(
                nodo.izquierda, profundidad + 1, f"{condicion_str}")
            self.imprimir_arbol(
                nodo.derecha, profundidad + 1, f"No {condicion_str}")


    def get_tree_structure(self, nodo=None):
        """
            Usa recursión para devolver la estructura completa de un árbol, incluyendo en cada
            nodo información relevante dependiendo si es un nodo hoja o un nodo de decisión que
            representa una regla/pregunta.
        """
        if nodo is None:
            nodo = self.raiz

        # Si es una hoja retorna la siguiente información.
        if nodo.es_hoja:

            # Se obtiene la impureza del nodo y se redondea a solo 3 decimales.
            numeroImpureza = round(nodo.impureza, 3)

            # Se comprueba que sea mayor a "-0.0" para establecer el valor en 0 si no es el caso.
            if numeroImpureza <= -0.0:
                numeroImpureza = 0
            
            # Se obtienen los valores unicos de las etiquetas con sus cantidades.
            etiquetasUnicas, cuenta = np.unique(nodo.etiquetas, return_counts=True)
            
            # Se comprueba la cantidad de valores para graficar diferente los values de cada nodo.
            valor = ""
            if self.etiquetas_originales[0] == etiquetasUnicas[0] and cuenta.size == 1:
                valor = f"[{cuenta[0]}, 0]"
            elif self.etiquetas_originales[0] != etiquetasUnicas[0] and cuenta.size == 1:
                valor = f"[0, {cuenta[0]}]"
            elif self.etiquetas_originales[0] == etiquetasUnicas[0] and cuenta.size == 2:
                valor = f"[{cuenta[0]}, {cuenta[1]}]"
            else:
                valor = f"[{cuenta[1]}, {cuenta[0]}]"

            # Guarda la información relevante del nodo.
            return {
                "tipo": "Hoja", 
                "criterio": f"{self.criterio}:{numeroImpureza}",
                "muestras": f"muestras: {nodo.muestras}",
                "valor": f"valor: {valor}",
                "clase": f"clase: {nodo.etiqueta}"
            }
        else:
            nombre_columna = self.nombres_caracteristicas[nodo.regla[0]]
            etiquetasUnicas, cuenta = np.unique(nodo.etiquetas, return_counts=True)

            # Devuelve la información relevante del nodo
            return {
                "tipo": "Decision",
                "reglaDescritiva": f"{nombre_columna} {nodo.regla[1]} {nodo.regla[2]}",
                "regla": f"{nodo.regla[0]} {nodo.regla[1]} {nodo.regla[2]}",
                "izquierda": self.get_tree_structure(nodo.izquierda), # Obtiene la estructura izquierda
                "derecha": self.get_tree_structure(nodo.derecha), # Obtiene la estructua derecha
                "criterio": f"{self.criterio}:{round(nodo.impureza, 3)}",
                "muestras": f"muestras: {nodo.muestras}",
                "valor": f"valor: [{cuenta[0]}, {cuenta[1]}]",
                "clase": f"clase: {nodo.etiqueta}",
            }
