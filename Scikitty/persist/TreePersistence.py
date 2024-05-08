import json
import sys
sys.path.append('..')  
from ..models.DecisionTree import Nodo

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
        if structure['tipo'] == 'Hoja':
            # Si el nodo es una hoja, crea un nodo de hoja con la etiqueta correspondiente
            #Obtener class
            # Obtener la regla de división del JSON
            rule_str = structure['regla']
            # Dividir la cadena en líneas
            lines = rule_str.strip().split('\n')
            clase_info = lines[3].strip().split()
            clase = clase_info[1]
            return Nodo(es_hoja=True, etiqueta=clase)
        else:
            nodo = self.separate_values(structure)
            # Crea un nodo de decisión con la regla
            # Reconstruye recursivamente los subárboles izquierdo y derecho
            nodo.izquierda = self._rebuild_tree(structure['izquierda'])
            nodo.derecha = self._rebuild_tree(structure['derecha'])
            return nodo  # Devuelve el nodo reconstruido (árbol reconstruido)
        
    def separate_values(self, structure):
        
            # Obtener la regla de división del JSON
            rule_str = structure['regla']
            # Dividir la cadena en líneas
            lines = rule_str.strip().split('\n')
            # Extraer la regla de división de la primera línea
            rule_info = lines[0].strip().split()
            index = rule_info[0]  # Índice
            operation = rule_info[1]  # Operación
            value = ' '.join(rule_info[2:])
            #Obtener la impureza
            impureza_info = lines[1].strip().split()
            impureza = impureza_info[1]
            #Obtener muestras
            muestras_info=lines[2].strip().split()
            muestras = muestras_info[1]
            #Obtener Etiquetas
            inicio_valor = rule_str.find("valor: [") + len("valor: [")
            fin_valor = rule_str.find("]", inicio_valor)

            # Extrae la cadena de valor
            cadena_valor = rule_str[inicio_valor:fin_valor]

            # Convierte la cadena de valor en una lista de enteros
            valores = [int(valor.strip()) for valor in cadena_valor.split(",")]
            #Obtener class
            clase_info = lines[4].strip().split()
            clase = clase_info[1]
            

            # Si el nodo es una decisión, extrae la regla de división y reconstruye el nodo
            # Suponiendo que 'rule' está almacenada como una cadena 'index == value'
            #index, operation, value = structure['regla'].split()
            rule = (index, operation, value)  # Convierte la cadena en una tupla de regla
            nodo = Nodo(es_hoja=False, regla=rule, etiqueta = clase, impureza=impureza, muestras=muestras, etiquetas = valores) 
            return nodo
