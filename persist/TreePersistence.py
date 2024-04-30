import json
sys.path.append('..') 
from models.DecisionTree import Node

class TreePersistence:
    def save_tree(self, tree, filename='tree_model.json'):
        tree_structure = tree.get_tree_structure()
        with open(filename, 'w') as file:
            json.dump(tree_structure, file, indent=4)

    def load_tree(self, filename='tree_model.json'):
        with open(filename, 'r') as file:
            tree_structure = json.load(file)
        return self._rebuild_tree(tree_structure)

    def _rebuild_tree(self, structure):
        if structure['type'] == 'Leaf':
            return Node(is_leaf=True, label=structure['label'])
        else:
            # Suponiendo que 'rule' es almacenada como una cadena 'index == value'
            index, operation, value = structure['rule'].split()
            rule = (int(index), operation, value)
            node = Node(is_leaf=False, rule=rule)
            node.left = self._rebuild_tree(structure['left'])
            node.right = self._rebuild_tree(structure['right'])
            return node
