from graphviz import Digraph

class TreeVisualizer:
    def __init__(self):
        self.dot = Digraph(comment='Decision Tree')

    def graph_tree(self, tree_structure, parent=None, edge_label=''):
        if tree_structure['type'] == 'Leaf':
            node_label = f"Leaf: {tree_structure['label']}"
            node_name = f"leaf_{id(tree_structure)}"
        else:
            node_label = tree_structure['rule']
            node_name = f"decision_{id(tree_structure)}"
            self.graph_tree(tree_structure['left'], parent=node_name, edge_label='True')
            self.graph_tree(tree_structure['right'], parent=node_name, edge_label='False')

        if parent:
            self.dot.node(node_name, label=node_label)
            self.dot.edge(parent, node_name, label=edge_label)
        else:
            self.dot.node(node_name, label=node_label, shape='box')

    def render(self, filename='tree', view=True):
        self.dot.render(filename, view=view, format='png')