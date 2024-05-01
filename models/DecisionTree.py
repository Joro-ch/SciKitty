import numpy as np
import pandas as pd


class Node:
    def __init__(self, is_leaf=False, rule=None, label=None, impurity=0, samples=0):
        self.is_leaf = is_leaf
        self.rule = rule
        self.label = label
        self.left = None
        self.right = None
        self.impurity = impurity
        self.samples = samples

    def __str__(self):
        return f"Leaf: {self.label}" if self.is_leaf else f"Rule: {self.rule}"


class DecisionTree:
    def __init__(self, features, labels, criterion='Entropy', min_samples_split=2, max_depth=None):
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.feature_names = features.columns.tolist() if isinstance(features, pd.DataFrame) else [f'Feature[{i}]' for i in range(np.array(features).shape[1])]
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self):
        self.root = self._build_tree(self.features, self.labels, 0)

    def _build_tree(self, features, labels, current_depth):
        if self._should_stop_splitting(labels, features.shape[0], current_depth):
            return Node(is_leaf=True, label=self._most_common_label(labels))

        best_rule, best_impurity = self._choose_best_rule(features, labels)
        if not best_rule:
            return Node(is_leaf=True, label=self._most_common_label(labels))

        left_indices, right_indices = self._split(features, best_rule)
        left_subtree = self._build_tree(features[left_indices], labels[left_indices], current_depth + 1)
        right_subtree = self._build_tree(features[right_indices], labels[right_indices], current_depth + 1)

        node = Node(rule=best_rule)
        node.left = left_subtree
        node.right = right_subtree
        node.impurity = best_impurity
        node.samples = features.size
        return node

    def _should_stop_splitting(self, labels, num_samples, current_depth):
        if len(np.unique(labels)) == 1 or num_samples < self.min_samples_split:
            return True
        if self.max_depth is not None and current_depth >= self.max_depth:
            return True
        return False

    def _most_common_label(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]

    def _choose_best_rule(self, features, labels):
        best_impurity = float('inf')
        best_rule = None
        n_samples = len(labels)
        feature_list = features.T

        for index, feature in enumerate(feature_list):
            unique_values = np.unique(feature)
            is_binary = len(unique_values) == 2
            is_categorical = isinstance(feature[0], str) and len(unique_values) > 2

            if not (is_binary or is_categorical):
                continue

            for value in unique_values:
                split_mask = feature == value
                split_labels = labels[split_mask]
                prob_value = len(split_labels) / n_samples
                impurity_value = self._calculate_impurity(split_labels)
                impurity = prob_value * impurity_value

                not_split_mask = feature != value
                not_split_labels = labels[not_split_mask]
                prob_not_value = len(not_split_labels) / n_samples
                not_impurity_value = self._calculate_impurity(not_split_labels)
                impurity += prob_not_value * not_impurity_value

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_rule = (index, '==', value)

        return best_rule, best_impurity

    def _split(self, features, rule):
        column_index, condition, value = rule
        left_indices = np.where(features[:, column_index] == value)[0]
        right_indices = np.where(features[:, column_index] != value)[0]
        return left_indices, right_indices

    def _calculate_impurity(self, labels):
        if self.criterion == 'Entropy':
            return self._calculate_entropy(labels)
        else:
            return self._calculate_gini(labels)

    def _calculate_entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _calculate_gini(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def predict(self, features):
        features = np.array(features)
        return [self._predict_single(feature, self.root) for feature in features]

    def _predict_single(self, feature, node):
        if node.is_leaf:
            return node.label
        if self._follow_rule(feature, node.rule):
            return self._predict_single(feature, node.left)
        else:
            return self._predict_single(feature, node.right)

    def _follow_rule(self, feature, rule):
        column_index, condition, value = rule
        return feature[column_index] == value if condition == '==' else feature[column_index] != value

    def print_tree(self, node=None, depth=0, condition="Root"):
        if node is None:
            node = self.root

        if node.is_leaf:
            print(f"{'|   ' * depth}{condition} -> Leaf: {node.label}")
        else:
            column_name = self.feature_names[node.rule[0]]
            condition_str = f"{column_name} {node.rule[1]} {node.rule[2]}"
            print(f"{'|   ' * depth}{condition} -> {condition_str}")
            self.print_tree(node.left, depth + 1, f"{condition_str}")
            self.print_tree(node.right, depth + 1, f"Not {condition_str}")

    def get_tree_structure(self, node=None):
        if node is None:
            node = self.root

        if node.is_leaf:
            return {"type": "Leaf", "label": node.label}
        else:
            column_name = self.feature_names[node.rule[0]]
            info = f"""
                {column_name} {node.rule[1]} {node.rule[2]}
                {self.criterion}: {node.impurity}
                samples: {node.samples}
                value: [{node.left.samples}, {node.right.samples}]
                class: {column_name}
            """

            return {
                "type": "Decision",
                "rule": info,
                "left": self.get_tree_structure(node.left),
                "right": self.get_tree_structure(node.right)
            }
