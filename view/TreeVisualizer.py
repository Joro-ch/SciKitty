from graphviz import Digraph

class TreeVisualizer:
    def __init__(self):
        # Inicializa un objeto Digraph de Graphviz para visualizar el árbol de decisión
        self.dot = Digraph(comment='Decision Tree')

    def graph_tree(self, tree_structure, parent=None, edge_label=''):
        # Método recursivo para graficar el árbol de decisión a partir de su estructura
        if tree_structure['type'] == 'Leaf':
            # Si es un nodo hoja, establece la etiqueta del nodo como "Leaf: <label>"
            node_label = f"Leaf: {tree_structure['label']}"
            node_name = f"leaf_{id(tree_structure)}"  # Nombre único para el nodo de hoja
        else:
            # Si es un nodo de decisión, establece la etiqueta del nodo como la regla de decisión
            node_label = tree_structure['rule']
            node_name = f"decision_{id(tree_structure)}"  # Nombre único para el nodo de decisión

            # Recursivamente grafica los subárboles izquierdo y derecho
            self.graph_tree(tree_structure['left'], parent=node_name, edge_label='True')
            self.graph_tree(tree_structure['right'], parent=node_name, edge_label='False')

        if parent:
            # Agrega el nodo actual al grafo y una conexión desde el nodo padre
            self.dot.node(node_name, label=node_label)
            self.dot.edge(parent, node_name, label=edge_label)
        else:
            # Si es la raíz del árbol, establece la forma del nodo como 'box'
            self.dot.node(node_name, label=node_label, shape='box')

    def render(self, filename='tree', view=True):
        # Renderiza el grafo como un archivo de imagen (por defecto, formato PNG)
        self.dot.render(filename, view=view, format='png')

# Explicación de uso:
# TreeVisualizer es una clase diseñada para visualizar árboles de decisión utilizando Graphviz.
# - El método __init__ inicializa un objeto Digraph de Graphviz con un comentario especificado.
# - El método graph_tree toma la estructura del árbol y genera un grafo que representa el árbol de manera recursiva.
# - El método render guarda el grafo como un archivo de imagen y lo muestra si se especifica 'view=True'.
