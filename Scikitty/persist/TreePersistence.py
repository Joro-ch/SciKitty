import json
import numpy as np
import sys
sys.path.append('..')  
from ..models.DecisionTree import Nodo

class TreePersistence:
    """
    Clase encargada de persistir el nodo en formato JSON para guardarlo,
    reconstruirlo o ser traducido a un programa de prolog.
    """
    @staticmethod
    def save_tree(tree, filename):
        """
        Serializa el árbol a un archivo JSON.
        """
        tree_dict = tree.get_tree()
        with open(filename, 'w') as f:
            json.dump(tree_dict, f, indent=4)
    
    @staticmethod
    def load_tree(filename):
        """
        Deserializa el archivo JSON a una estructura de árbol.
        """
        with open(filename, 'r') as f:
            tree_dict = json.load(f)
        
        def _reconstruir_nodo(nodo_dict):
            nodo = Nodo(
                es_hoja=nodo_dict['es_hoja'],
                regla=nodo_dict.get('regla'),
                etiqueta=nodo_dict.get('etiqueta'),
                impureza=nodo_dict['impureza'],
                etiquetas=np.array(nodo_dict['etiquetas'])
            )
            if not nodo.es_hoja:
                nodo.izquierda = _reconstruir_nodo(nodo_dict['izquierda'])
                nodo.derecha = _reconstruir_nodo(nodo_dict['derecha'])
            return nodo

        return _reconstruir_nodo(tree_dict)