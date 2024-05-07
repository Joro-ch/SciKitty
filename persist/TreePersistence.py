import json
import sys
sys.path.append('..')  
from models.DecisionTree import nodo 

class TreePersistence:
    """
    Clase encargada de persistir el nodo en formato JSON para guardarlo,
    reconstruirlo o ser traducido a un programa de prolog.
    """
    def save_tree(self, tree, filename='tree_model.json'):
        """
        Guarda la estructura del árbol en un archivo JSON
        """
        tree_structure = tree.get_tree_structure()  # Obtiene la estructura del árbol como un diccionario
        with open(filename, 'w') as file:
            json.dump(tree_structure, file, indent=4)  # Escribe la estructura del árbol en el archivo JSON con formato indentado

    def load_tree(self, filename='tree_model.json'):
        """
        Carga la estructura del árbol desde un archivo JSON y reconstruye el árbol
        """
        with open(filename, 'r') as file:
            tree_structure = json.load(file)  # Lee la estructura del árbol desde el archivo JSON
        return self._rebuild_tree(tree_structure)  # Reconstruye el árbol a partir de la estructura cargada

    def _rebuild_tree(self, structure):
        """
        Reconstruye el árbol a partir de una estructura de árbol dada (en forma de diccionario)
        """
        if structure['type'] == 'Leaf':
            # Si el nodo es una hoja, crea un nodo de hoja con la etiqueta correspondiente
            return nodo(is_leaf=True, label=structure['label'])
        else:
            # Si el nodo es una decisión, extrae la regla de división y reconstruye el nodo
            # Suponiendo que 'rule' está almacenada como una cadena 'index == value'
            index, operation, value = structure['rule'].split()
            rule = (int(index), operation, value)  # Convierte la cadena en una tupla de regla
            nodo = nodo(is_leaf=False, rule=rule)  # Crea un nodo de decisión con la regla
            # Reconstruye recursivamente los subárboles izquierdo y derecho
            nodo.left = self._rebuild_tree(structure['left'])
            nodo.right = self._rebuild_tree(structure['right'])
            return nodo  # Devuelve el nodo reconstruido (árbol reconstruido)

"""
Explicación de uso:
TreePersistence es una clase diseñada para guardar y cargar árboles de decisión en formato JSON.
La función save_tree toma un objeto de árbol (tree) y lo guarda en un archivo JSON con el nombre especificado (por defecto 'tree_model.json').
La función load_tree carga un árbol desde un archivo JSON previamente guardado y devuelve el árbol reconstruido.
La función _rebuild_tree es un método privado que reconstruye el árbol a partir de una estructura de árbol dada en formato de diccionario.
"""
