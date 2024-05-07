# Scikitty

### 1) Descripción del sprint #1 del proyecto
En el siguiente sprint se podrán observar la primera parte de la implementación y la creación del modelo de árbol de decisión para variables categóricas binarias. En donde se busca poder entrenar y regular (por medio de hiperparámetros) al modelo por medio de un dataset y mediante de variables de testeo poder observar las decisiones tomadas por medio del árbol

### 2) Ambiente:
Este proyecto se deberá de correr desde el prompt Anaconda, en donde se deberá de tener instalado las librerías numpy, pandas y graphviz. Desde el prompt de anaconda se podrán instalarlos:
```
pip install numpy
pip install pandas
pip install graphviz
```
### 3) Dataset

En este proyecto se usando tres datasets en específico:
1. **fictional_disease.csv:** En este dataset se busca poder predecir si una persona es enferma o no, por medio de característica como la edad, el género y su historial de fumador.

2. **fictional_reading_place.csv:** Para este dataset se intenta predecir si un lector lee o se salta una lectura, según características como el autor, hilo, longitud y el lugar de lectura. 

3. **playTennis.csv:** Con este dataset se busca saber si se juega tennis por medio de características como el clima, la temperatura, la humedad y el viento.

Sin embargo para este proyecto sólo se tomarán las características categóricas binarias.

### 4) ¿Cómo ejecutar el proyecto?
Para poder ejecuta cada uno de los scripts, es necesario seguir cada uno de los siguientes pasos:
* Abrir Anaconda prompt
* Se debe de ubicar en el directorio 'scripts' del proyecto, en donde se encuentran los scripts correspondiente a cada uno de los dataset. Para poder ejecutar cada uno de ellos, es necesario colocar los siguientes comandos:

* Para poder ejecutar el árbol de decisión de fictional_disease.csv: 
```
python fictional_disease.py
```
* Para poder ejecutar el árbol de decisión de fictional_reading_place.csv 
```
python fictional_reading_place.py
```
* Para poder ejecutar el árbol de decisión de playTennis.csv: 
```
python playTennis.py
```
En el caso de poder compararlo con las salidas de la librería de Scikit Learn, puede ejecutar los siguientes scripts,
para comparar los resultados de las dos librerias:

* Para poder ejecutar el resultado de scikitlearn del dataset fictional_disease.csv: 
```
python fictional_disease-scikitlearn.py
```
* Para poder ejecutar el resultado de scikitlearn del dataset fictional_reading_place.csv:
```
python fictional_reading_place-scikitlearn.py
```
* Para poder ejecutar el resultado de scikitlearn del dataset playTennis.csv:
```
python playTennis-scikitlearn.py
```

### 5) Salidas:
Una vez ejecutado el archivo se mostrarán el resultado de los datos de prueba y los resultados reales para poder comparar la predicción del árbol y también la librería graphviz creará una representación del árbol de decisión que respeta los hiperparámetros
*** Se debe de  agregar las metrics?

### 6) Hiperparámetros
En caso de querer poder cambiar los hiperparámetros, se tendrá que modificar directamente el archivo a ejecutar en donde se podrán reconocer las variables:
* **max_depth:** Recibirá el número máximo de profundidad que contendrá el árbol de decisión
* **min_samples_split:** Recibirá el número mínimo de ejemplares que debe de tener un nodo para poder generar un split
Estos hiperparámetros se tendrán que remplazar el símbolo '?' y colocar un número. Con esto se podrán ajustar al generar e instanciar el árbol de decisión:

```
	dt = DecisionTree(X_train, y_train, criterion='Entropy', min_samples_split=?, max_depth=?)
	
```

##Estudiantes
- Axel Monge Ramírez
- Andrel Ramírez Solis
- John Rojas Chinchilla
- Abigail Salas Ramírez