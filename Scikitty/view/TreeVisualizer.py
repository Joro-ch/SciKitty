import graphviz

class TreeVisualizer:
    def __init__(self):
        """
            Inicializa un objeto Graphviz Digraph para visualizar el árbol de decisión
        """
        self.grafo = graphviz.Digraph(format='png')

    def graph_tree(self, estructura_arbol, padre=None, etiqueta_arista='', nivel=0, posicion=0):
        """
            Método recursivo para graficar el árbol de decisión a partir de su estructura utilizando Graphviz
        """
        if estructura_arbol['tipo'] == 'Hoja':
            nombre_nodo = f"hoja_{id(estructura_arbol)}"
            color = '#ffa500'  # Un naranja brillante y llamativo
        else:
            nombre_nodo = f"decision_{id(estructura_arbol)}"
            color = '#5a9ad5'  # Un azul cielo claro
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

        etiqueta_nodo = self.formato_etiqueta(estructura_arbol)
        self.grafo.node(nombre_nodo, label=etiqueta_nodo, shape='box', style='filled', fillcolor=color)
        if padre:
            self.grafo.edge(padre, nombre_nodo, label=etiqueta_arista)

    def formato_etiqueta(self, nodo):
        """
            Crea una etiqueta formateada con los detalles del nodo centrados y en líneas separadas.
            Divide el texto de 'regla' en líneas individuales y las muestra como etiquetas del nodo.
        """
        regla_completa = nodo.get('regla', "")
        lineas_regla = regla_completa.strip().split('\n')
        lineas_limpas = [linea.strip() for linea in lineas_regla if linea.strip()]
        etiqueta_nodo = "\n".join(lineas_limpas)
        return etiqueta_nodo

    def get_graph(self, nombre_archivo='arbol', ver=True):
        """
            Renderiza el grafo como un archivo de imagen y muestra el grafo.
        """
        self.grafo.render(nombre_archivo, view=ver)