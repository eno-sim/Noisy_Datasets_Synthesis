import numpy as np
import pandas as pd




class DataPreprocessing:
    
    def __init__(self, path, numpy=True):
        self._data, self._labels, self._feature_names = self._load_data(path, numpy)

    def _load_data(self, path, numpy=True):
        """
        Load data from a CSV file, preprocess it, and identify labels if present.

        Parameters:
        path (str): The path to the CSV file.
        numpy (bool): Whether to return the data as a NumPy array or a DataFrame. Default is True.

        Returns:
        tuple: (data, labels, feature_names)
        """
        try:
            data = pd.read_csv(path)
        except FileNotFoundError:
            raise Exception(f"File not found at path: {path}")

        # Check if the last column is binary (0 or 1 only) and use it as labels
        last_column = data.iloc[:, -1]
        if set(last_column.unique()) == {0, 1}:
            labels = last_column.to_numpy()
            data = data.iloc[:, :-1]
        else:
            labels = None

        # Remove categorical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        data = data.drop(columns=categorical_columns)

        feature_names = data.columns.tolist()

        if numpy:
            data = data.to_numpy()

        return data, labels, feature_names

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def feature_names(self):
        return self._feature_names    

    def noise_level_adjustment(self, threshold=0.1):
        """
        Adjust the noise level in the instance's data and labels to meet the specified threshold.
        
        Parameters:
        threshold (float): The desired noise level threshold. Default is 0.1.
        
        Updates:
        self._data and self._labels with adjusted noise levels.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")
        
        if self._labels is None:
            raise ValueError("No labels found in the data.")
        
        clean_data = self._data[self._labels == 0]
        current_noise_count = self._data[self._labels == 1].shape[0]
        target_noise_count = int((threshold * clean_data.shape[0]) / (1 - threshold))
        noise_to_remove_count = current_noise_count - target_noise_count
        
        noise_indices = np.where(self._labels == 1)[0]
        if noise_to_remove_count > 0:
            noise_indices_to_remove = np.random.choice(noise_indices, size=noise_to_remove_count, replace=False)
            self._data = np.delete(self._data, noise_indices_to_remove, axis=0)
            self._labels = np.delete(self._labels, noise_indices_to_remove)

    def normalize_features(self):
        """
        Normalize the features using StandardScaler.
        """
        self._data = self._scaler.fit_transform(self._data)

    def __str__(self):
        return f"DataPreprocessing(n_samples={self._data.shape[0]}, n_features={self._data.shape[1]})"
