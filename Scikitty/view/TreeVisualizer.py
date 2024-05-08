import graphviz

class TreeVisualizer:
    """
    Clase para visualizar un árbol de decisión utilizando Graphviz.
    """
    def __init__(self):
        """
        Inicializa un objeto Graphviz Digraph para visualizar el árbol de decisión.
        """
        self.grafo = graphviz.Digraph(format='png')

    def graph_tree(self, estructura_arbol, padre=None, etiqueta_arista='', nivel=0, posicion=0):
        """
        Método recursivo para graficar el árbol de decisión a partir de su estructura utilizando Graphviz.

        Parámetros:
        estructura_arbol: dict, La estructura del árbol de decisión definiendo cada nodo y los atributos a mostrar por nodo.
        padre: str, El nodo padre en el grafo.
        etiqueta_arista: str, La etiqueta de la arista que conecta al nodo padre con el nodo actual. (Se usa true y false predeterminadamente).
        nivel: int, El nivel del nodo en el árbol.
        posicion: int, La posición del nodo en el nivel.

        Funcionalidad:
        Dibuja el nodo actual y llama recursivamente a sí mismo para dibujar los nodos hijos.
        """
        # Dibuja un nodo usando informacion relevante de estructura_arbol según si es hoja o regla.
        if estructura_arbol['tipo'] == 'Hoja':
            nombre_nodo = f"hoja_{id(estructura_arbol)}"
            color = '#ffa500' 
        else:
            nombre_nodo = f"decision_{id(estructura_arbol)}"
            color = '#5a9ad5' 
            self.graph_tree(estructura_arbol['izquierda'], 
                            padre=nombre_nodo, 
                            etiqueta_arista='True', 
                            nivel=nivel+1, 
                            posicion=posicion-1)
            self.graph_tree(estructura_arbol['derecha'], 
                            padre=nombre_nodo, 
                            etiqueta_arista='False', 
                            nivel=nivel+1, 
                            posicion=posicion+1)

        etiqueta_nodo = self.formato_etiqueta(estructura_arbol) # LLama al método que formatea la informació a desplegar en el nodo.
        self.grafo.node(nombre_nodo, label=etiqueta_nodo, shape='box', style='filled', fillcolor=color)
        if padre:
            self.grafo.edge(padre, nombre_nodo, label=etiqueta_arista)

    def formato_etiqueta(self, nodo):
        """
        Crea una etiqueta formateada con los detalles del nodo centrados y en líneas separadas.
        Divide el texto de 'regla' en líneas individuales y las muestra como etiquetas del nodo.

        Parámetros:
        nodo: dict, La información del nodo a formatear.

        Retorna:
        str: La etiqueta formateada para el nodo.
        """
        # Formatea la información a desplegar en el nodo según si es hoja o, si no es hoja, es regla.
        tipo = nodo.get('tipo', "")
        if tipo == 'Hoja':
            criterio = nodo.get('criterio', "")
            muestras = nodo.get('muestras', "")
            valor = nodo.get('valor', "")
            clase = nodo.get('clase', "")
            etiqueta_nodo = f"{criterio}\n{muestras}\n{valor}\n{clase}"
        else:
            regla = nodo.get('reglaDescritiva', "")
            criterio = nodo.get('criterio', "")
            muestras = nodo.get('muestras', "")
            valor = nodo.get('valor', "")
            clase = nodo.get('clase', "")
            etiqueta_nodo = f"{regla}\n{criterio}\n{muestras}\n{valor}\n{clase}"
        return etiqueta_nodo

    def get_graph(self, nombre_archivo='arbol', ver=True):
        """
        Renderiza el grafo como un archivo de imagen y muestra el grafo.

        Parámetros:
        nombre_archivo: str, El nombre del archivo de imagen a crear.
        ver: bool, Si es True, abre el archivo de imagen después de crearlo.

        Funcionalidad:
        Renderiza el grafo en un archivo de imagen con el nombre especificado y, si se indica, abre el archivo de imagen.
        """
        self.grafo.render(nombre_archivo, view=ver) # Método de Digraph que se utiliza para generar (o "renderizar") y guardar el grafo creado en un archivo de salida
        # Usamos ver como true por defecto para abrir el archivo, eso lo mandeja Digraph.

"""
Modo de uso:

# Obtener la estructura del árbol
tree_structure = dt.get_tree_structure()

# Crear un visualizador del árbol
visualizer = TreeVisualizer()

# Graficar el árbol
visualizer.graph_tree(tree_structure)

# Renderizar y mostrar el grafo del árbol
visualizer.get_graph(f'{file_name}_tree', ver=True)
"""
