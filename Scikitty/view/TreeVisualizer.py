import matplotlib.pyplot as plt
import networkx as nx

class TreeVisualizer:
    def __init__(self):
        """
            Inicializa un grafo dirigido de NetworkX para visualizar el árbol de decisión
        """
        self.grafo = nx.DiGraph()
        self.posiciones = {}

    def graph_tree(self, estructura_arbol, padre=None, etiqueta_arista='', nivel=0, posicion=0, distancia_entre_nodos=2, distancia_vertical=1):
        """
            Método recursivo para graficar el árbol de decisión a partir de su estructura, con posicionamiento específico para evitar solapamientos
        """
        if estructura_arbol['tipo'] == 'Hoja':
            # Si es un nodo hoja, establece la etiqueta del nodo como "Hoja: <etiqueta>"
            etiqueta_nodo = estructura_arbol['regla']
            nombre_nodo = f"hoja_{id(estructura_arbol)}"
            color = 'lightgreen'  # Color claro para las hojas
        else:
            # Si es un nodo de decisión, establece la etiqueta del nodo como la regla de decisión
            etiqueta_nodo = estructura_arbol['regla']
            nombre_nodo = f"decision_{id(estructura_arbol)}"
            color = 'lightblue'  # Color claro para los nodos de decisión

            # Recursivamente grafica los subárboles izquierdo y derecho con posiciones ajustadas
            self.graph_tree(estructura_arbol['izquierda'], 
                            padre=nombre_nodo, 
                            etiqueta_arista='True', 
                            nivel=nivel+1, 
                            posicion=posicion-distancia_entre_nodos, 
                            distancia_entre_nodos=distancia_entre_nodos/1.5, 
                            distancia_vertical=distancia_vertical)
            self.graph_tree(estructura_arbol['derecha'], 
                            padre=nombre_nodo, 
                            etiqueta_arista='False', 
                            nivel=nivel+1, 
                            posicion=posicion+distancia_entre_nodos, 
                            distancia_entre_nodos=distancia_entre_nodos/1.5, 
                            distancia_vertical=distancia_vertical)

        # Calcula la posición del nodo actual y lo agrega al grafo
        x_pos = posicion
        y_pos = -nivel * distancia_vertical
        self.posiciones[nombre_nodo] = (x_pos, y_pos)
        self.grafo.add_node(nombre_nodo, label=etiqueta_nodo, color=color)
        if padre:
            self.grafo.add_edge(padre, nombre_nodo, label=etiqueta_arista)

    def get_graph(self, nombre_archivo='arbol', ver=True):
        """
            Renderiza el grafo como un archivo de imagen (por defecto, formato PNG) y muestra el grafo con nodos cuadrados y posicionamiento vertical específico
        """
        colors = [n[1]['color'] for n in self.grafo.nodes(data=True)]
        labels = {n[0]:n[1]['label'] for n in self.grafo.nodes(data=True)}
        edge_labels = nx.get_edge_attributes(self.grafo, 'label')

        fig, ax = plt.subplots(figsize=(15, 10))  # Ajusta el tamaño de la figura
        nx.draw(self.grafo, self.posiciones, labels=labels, with_labels=True, node_color=colors, node_size=2000, edge_color='black', font_size=10, font_color='black', ax=ax, node_shape='s')
        nx.draw_networkx_edge_labels(self.grafo, self.posiciones, edge_labels=edge_labels, font_color='red', ax=ax)

        plt.axis('off')  # Desactiva los ejes para una visualización más limpia
        plt.tight_layout()
        plt.savefig(f"{nombre_archivo}.png")  # Guarda la imagen
        if ver:
            plt.show()  # Muestra el gráfico
