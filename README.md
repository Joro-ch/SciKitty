# Scikitty

### 1) Descripción del sprint #1 del proyecto
En el siguiente sprint se podrán observar la primera parte de la implementación y la creación del modelo de árbol de decisión para variables categóricas binarias. En donde se busca poder entrenar y regular (por medio de hiperparámetros) al modelo por medio de un dataset y mediante de variables de testeo poder observar las decisiones tomadas por medio del árbol, además de poder serializar el árbol en un documento .json. 

### 2) Ambiente:
Este proyecto se deberá de correr desde el prompt Anaconda, en donde se deberá de tener instalado las librerías numpy, pandas y graphviz. Desde el prompt de anaconda se podrán instalarlos mediante las siguientes instrucciones:
```
pip install numpy
pip install pandas
conda install graphviz
```
### 3) Datasets

En este proyecto se estarán utilizando diferentes datasets, pero para temas del primer seguimiento se estarán utilizando tres en específicos:
1. **fictional_disease.csv:**
* **Target:** Disease.
* **Características:** Age, Gender, SmokerHistory.
El modelo busca predecir si una persona es enferma o no, por medio de características como la edad, el género y su historial de fumador.

2. **fictional_reading_place.csv:** 
* **Target:** User action.
* **Características:** Author, Thread, Length, Where read.
El modelo busca predecir si un lector lee o se salta una lectura, según características como el autor, hilo, longitud y el lugar de lectura. 

3. **playTennis.csv:** 
* **Target:** Play Tennis
* **Características:** Outlook, Temperature, Humidity, Wind
El modelo busca predecir si una persona juega o no tennis por medio de características como el pronóstico del clima, la temperatura, la humedad y el viento.

El modelo creado para este proyecto tendrá la capacidad de poder validar las variables categóricas binarios, multiclase y continuas. Por lo que sí tomará en cuenta estas variables de los dataset y podrá tomar las desiciones en base a estas. En consecuencia se puede tomar en cuenta el siguiente dataset:

1. **CO2_car_emision.csv:**
* **Target:** CO2
* **Características:** Car, Model, Volume, Weight
El modelo busca predecir la cantidad de CO2 que genera un automóvil, según marca, modelo, volumen y peso del automóvil.

### 4) ¿Cómo ejecutar el proyecto?
Para poder ejecuta cada uno de los scripts, es necesario seguir cada uno de los siguientes pasos:
1. Abrir Anaconda prompt
2. Se debe de ubicar en el directorio Scikitty en la caperta 'demos' utilizando el comando cd, en donde se encuentran los scripts correspondiente a cada uno de los dataset. Para poder ejecutar cada uno de ellos, es necesario colocar los siguientes comandos:
3. Para poder ejecutar el modelo creado según el dataset a escoger tendrá que ejecutar las siguientes instrucciones:
* Para poder ejecutar el árbol de decisión de **fictional_disease.csv: **
```
python fictional_disease.py
```
* Para poder ejecutar el árbol de decisión de **fictional_reading_place.csv** 
```
python fictional_reading_place.py
```
* Para poder ejecutar el árbol de decisión de **playTennis.csv: **
```
python playTennis.py
```
* Para poder ejecutar el árbol de decisión de **CO2_car_emision.csv**
```
python CO2_car_emision.py
```
En el caso de poder compararlo con las salidas de la librería de **Scikit Learn**, puede ejecutar los siguientes scripts, para comparar los resultados de las dos librerias:

1. Para poder ejecutar el resultado de **scikitlearn del dataset fictional_disease.csv: **
```
python fictional_disease-scikitlearn.py
```
2. Para poder ejecutar el resultado de *scikitlearn* del dataset **fictional_reading_place.csv:**
```
python fictional_reading_place-scikitlearn.py
```
3. Para poder ejecutar el resultado de *scikitlearn* del dataset **playTennis.csv:**
```
python playTennis-scikitlearn.py
```
4. Para poder ejecutar el árbol de decisión de *scikitlearn* del dataset **CO2_car_emision.csv**
```
python CO2_car_emision-scikitlearn.py
```

### 5) Salidas:
Con respecto de los resultados de las métricas se utilizarán las de la librería de Sklearn.metrics. Para visualizar el árbol se usará la clase TreeVisualizer, el cual se encargará de generar una imágen .png, con ayuda de la librería Graphviz

**SCI-KITTY: **
Se mostrará los datos del árbol entrenado original y una vez que se cree el archivo JSON se recuperará y se guardará el modelo entrenado en un nuevo árbol, en donde se mostrarán y se compararán los siguientes datos del árbol original y del recuperado:
* **Exactitud:**Muestra las etiquetas que han predicho correctamente.
* **Precisión:** Muestra las etiquetas predichas positivas que son correcta.
* **Recall:**Muestra las etiquetas positivas reales que se han predicho correctamente.
* **F1-score:** Muestra la predicción que tuvo el árbol en la fase de prueba
* **Matriz de confusión: ** Se mostrará la matriz de confusión.
* **Etiquetas predichas: **Se mostrará cuáles fueron las etiquetas que predijo el modelo.
* **Etiquetas reales:** Se mostrará cuáles son las etiquetas reales. Con el fin de poder compararlas con las etiquetas predichas.
* **Visualización del árbol:** Para poder visualizar el árbol entrenado original
* **Archivo JSON:** Se crea un archivo JSON en donde se guardarán los nodos con información relevante para poder recontruir el árbol, como lo es:
1. Es_hoja: True/ False
2. Regla: [índice, condición, valor ]
3. La etiqueta
4. Impureza
5. Etiquetas: Los valores de las muestras
6. La cantidad de muestras
* **Visaulización del árbol recuperado:** Se visualizará el árbol recuperado por medio del archivo JSON

**SCI-KIT**
Se mostrarán los mismos datos que en el caso de SCI-KITTY. Para visualizar el árbol se usará la librería de Matplotlib

### 6) Hiperparámetros
En caso de querer poder cambiar los hiperparámetros, se tendrá que modificar directamente el archivo a ejecutar en donde se podrán reconocer las variables:
* **max_depth:** Recibirá el número máximo de profundidad que contendrá el árbol de decisión
* **min_samples_split:** Recibirá el número mínimo de ejemplares que debe de tener un nodo para poder generar un split
Estos hiperparámetros se tendrán que remplazar el símbolo '?' y colocar un número. Con esto se podrán ajustar al generar e instanciar el árbol de decisión:

```
dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=?, max_depth=?)
	
```

##Estudiantes
- Axel Monge Ramírez
- Andrel Ramírez Solis
- John Rojas Chinchilla
- Abigail Salas Ramírez