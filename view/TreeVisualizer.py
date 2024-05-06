from graphviz import Digraph

class VisualizadorArbol:
    def __init__(self):
        # Inicializa un objeto Digraph de Graphviz para visualizar el árbol de decisión
        self.grafo = Digraph(comment='Árbol de Decisión')

    def graficar_arbol(self, estructura_arbol, padre=None, etiqueta_arista=''):
        # Método recursivo para graficar el árbol de decisión a partir de su estructura
        if estructura_arbol['tipo'] == 'Hoja':
            # Si es un nodo hoja, establece la etiqueta del nodo como "Hoja: <etiqueta>"
            etiqueta_nodo = f"Hoja: {estructura_arbol['etiqueta']}"
            nombre_nodo = f"hoja_{id(estructura_arbol)}"  # Nombre único para el nodo de hoja
        else:
            # Si es un nodo de decisión, establece la etiqueta del nodo como la regla de decisión
            etiqueta_nodo = estructura_arbol['regla']
            nombre_nodo = f"decision_{id(estructura_arbol)}"  # Nombre único para el nodo de decisión

            # Recursivamente grafica los subárboles izquierdo y derecho
            self.graficar_arbol(estructura_arbol['izquierda'], padre=nombre_nodo, etiqueta_arista='Verdadero')
            self.graficar_arbol(estructura_arbol['derecha'], padre=nombre_nodo, etiqueta_arista='Falso')

        if padre:
            # Agrega el nodo actual al grafo y una conexión desde el nodo padre
            self.grafo.node(nombre_nodo, label=etiqueta_nodo)
            self.grafo.edge(padre, nombre_nodo, label=etiqueta_arista)
        else:
            # Si es la raíz del árbol, establece la forma del nodo como 'box'
            self.grafo.node(nombre_nodo, label=etiqueta_nodo, shape='box')

    def renderizar(self, nombre_archivo='arbol', ver=True):
        # Renderiza el grafo como un archivo de imagen (por defecto, formato PNG)
        self.grafo.render(nombre_archivo, view=ver, format='png')


# Explicación de uso:
# TreeVisualizer es una clase diseñada para visualizar árboles de decisión utilizando Graphviz.
# - El método __init__ inicializa un objeto Digraph de Graphviz con un comentario especificado.
# - El método graph_tree toma la estructura del árbol y genera un grafo que representa el árbol de manera recursiva.
# - El método render guarda el grafo como un archivo de imagen y lo muestra si se especifica 'view=True'.
