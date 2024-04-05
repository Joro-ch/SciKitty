import pandas as pd
import numpy as np

class DecisionTree:

    def __init__(self, file):
        self.data = self.__init_data(file)

    def __init_data(self, file) -> pd.DataFrame:
        """
            Get a file's directory to return his corresponding DataFrame without have to know the file's type.
        """

        file_extension = file.split(".")[-1]
        types = {
            'csv': pd.read_csv,
            'xlsx': pd.read_excel,
            'xls': pd.read_excel,
            'h5': pd.read_hdf,
            'hdf5': pd.read_hdf,
            'json': pd.read_json,
            'html': pd.read_html,
            'htm': pd.read_html
        }

        if file_extension in types:
            return types[file_extension](file)
        else:
            raise ValueError("Unsupported file format")

    def entropy(self, target_column):
        # Get the elements of the column and their total count
        elements, counts = np.unique(target_column, return_counts=True)
        # Calculate entropy based on its formula
        result = -np.sum([(counts[i]/np.sum(counts)) *
                    np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return result

    # Calculate the Gini index
    def gini(self, target_column):
        # Get the elements of the column and their total count
        elements, counts = np.unique(target_column, return_counts=True)
        # Calculate the Gini index based on its formula
        result = 1 - \
            np.sum([(counts[i]/np.sum(counts)) **
                    2 for i in range(len(elements))])
        return result

    # Calculate impurity for a given feature
    def calculate_impurity(self, split_feature, target_feature, model='entropy'):
        # Get the elements of the current split_feature
        elements = self.data[split_feature].unique()
        model_split_result = 0
        # Calculate impurity criterion for each element of split_feature
        for element in elements:
            # Get the new subset
            subset = self.data[self.data[split_feature] == element]
            # Calculate impurity criterion for the subset based on the selected model
            if model == 'entropy':
                model_subset = self.entropy(subset[target_feature])
            elif model == 'gini':
                model_subset = self.gini(subset[target_feature])
            # Determine the cumulative sum of weighted averages for the subsets
            model_split_result += (len(subset) / len(self.data)) * model_subset
        return model_split_result

    def get_min_impurity(self, target_feature, model='entropy'):
        features = list(self.data)
        if len(features) > 0: features.pop(0)
        if target_feature in features: features.remove(target_feature)
        
        # For each feature, calculate its impurity based on the model and the target_feature
        for feature in features:
            results = self.calculate_impurity(feature, target_feature, model)
            print(f'{model.capitalize()} criterion for {feature}: {results}')


target_feature = 'mamifero'
tree = DecisionTree("mamiferos.csv")
tree.get_min_impurity(target_feature, model="entropy")
