import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
import itertools
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from pyod.models.ocsvm import OCSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from hog_bisect.bisect import BisectHOGen
import pyod
from pymfe.mfe import MFE





proportions_list = [ [0.1, 0, 0.9], [0.2, 0, 0.8],
                     [0.3, 0, 0.7], [0.4, 0, 0.6],
                     [0.2, 0.1, 0.7], [0.3, 0.1, 0.6],
                     [0.2, 0.2, 0.6],
                     [0.8, 0, 0.2], [0.9, 0, 0.1], 
                     [0.7, 0, 0.3], [0.6, 0, 0.4], 
                     [0.5, 0, 0.5], 
                     [0.7, 0.1, 0.2], 
                     [0.6, 0.1, 0.3],
                     [0.5, 0.1, 0.4], [0.4, 0.1, 0.5],
                     [0.6, 0.2, 0.2], 
                     [0.5, 0.2, 0.3], [0.5, 0.2, 0.3],
                     [0.4, 0.2, 0.4], 
                     [0.7, 0.3, 0], [1, 0, 0],
                     [0, 0, 1], [0.9, 0.1, 0],
                     [0.8, 0.2, 0]  ]


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
     
     

class AnomalyGenerator:
    def __init__(self, data):
        """
        Initialize the AnomalyGenerator with the given data.

        Args:
            data (np.ndarray): The input dataset.
        """
        self.data = data

    def fit_validators(self, lof_neighbors=10, iso_contamination=0.05, svm_nu=0.1, svm_gamma=0.1, svm_kernel="rbf"):
        """
        Fits the three anomaly detection models (LOF, Isolation Forest, One-Class SVM) to the data.

        Args:
            lof_neighbors (int): Number of neighbors for LocalOutlierFactor.
            iso_contamination (float): Contamination factor for IsolationForest.
            svm_nu (float): Nu parameter for OneClassSVM.
            svm_gamma (float): Gamma parameter for OneClassSVM.
            svm_kernel (str): Kernel type for OneClassSVM.

        Returns:
            tuple: Fitted LOF, Isolation Forest, and One-Class SVM models.
        """
        clf_lof = LocalOutlierFactor(n_neighbors=lof_neighbors, novelty=True)
        clf_iso = IsolationForest(contamination=iso_contamination, random_state=42)
        clf_svm = OneClassSVM(nu=svm_nu, kernel=svm_kernel, gamma=svm_gamma)

        clf_lof.fit(self.data)
        clf_iso.fit(self.data)
        clf_svm.fit(self.data)

        return clf_lof, clf_iso, clf_svm

    def validated_samples(self, clf_lof, clf_iso, clf_svm, samples):
        """
        Validates samples using pre-fitted LOF, Isolation Forest, and One-Class SVM models.

        Args:
            clf_lof (LocalOutlierFactor): Fitted Local Outlier Factor model.
            clf_iso (IsolationForest): Fitted Isolation Forest model.
            clf_svm (OneClassSVM): Fitted One-Class SVM model.
            samples (np.ndarray): Array of samples to validate.

        Returns:
            tuple: A tuple containing three NumPy arrays:
                - easy_outliers: Samples identified as outliers by all three methods.
                - medium_outliers: Samples identified by two out of three methods.
                - interesting_outliers: Samples identified by only one method.
        """
        lof_predictions = clf_lof.predict(samples)
        iso_predictions = clf_iso.predict(samples)
        svm_predictions = clf_svm.predict(samples)

        all_predictions = np.vstack([lof_predictions, iso_predictions, svm_predictions])
        
        easy_outliers = samples[(all_predictions == -1).all(axis=0)]
        medium_outliers = samples[((all_predictions == -1).sum(axis=0) == 2)]
        interesting_outliers = samples[((all_predictions == -1).sum(axis=0) == 1)]

        return easy_outliers, medium_outliers, interesting_outliers

    def generate_noise(self, *args, **kwargs):
        """
        Abstract method to be implemented by subclasses for specific noise generation.
        """
        raise NotImplementedError("Subclasses must implement generate_noise method")

    def add_noise(self, *args, **kwargs):
        """
        Abstract method to be implemented by subclasses for adding noise to the dataset.
        """
        raise NotImplementedError("Subclasses must implement add_noise method")
    

    
class GlobalAnomalyGenerator(AnomalyGenerator):
    def __init__(self, data):
        super().__init__(data)

    def generate_point_anomaly(self, n_anomalies=1, mean=0, std=0.1):
        """
        Generates a point anomaly by selecting a random point and adding Gaussian noise to it.

        Args:
            n_anomalies (int): Number of anomalies to generate.
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.

        Returns:
            np.ndarray: Generated anomalies.
        """
        anomalies = []
        for _ in range(n_anomalies):
            idx = np.random.randint(0, self.data.shape[0])
            anomaly = self.data[idx] + np.random.normal(mean, std, self.data.shape[1])
            anomalies.append(anomaly)
        return np.array(anomalies)

    def gaussian_noise(self, value, mean, std):
        """
        Adds Gaussian noise to a value.

        Args:
            value (float): The original value.
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.

        Returns:
            float: The value with added Gaussian noise.
        """
        return value + np.random.normal(mean, std)

    def uniform_noise(self, value, lower, upper):
        """
        Adds uniform noise to a value.

        Args:
            value (float): The original value.
            lower (float): Lower bound of the uniform distribution.
            upper (float): Upper bound of the uniform distribution.

        Returns:
            float: The value with added uniform noise.
        """
        return value + np.random.uniform(lower, upper)

    def add_global_noise(self, pct_data, pct_features, noise_types, gaussian_mean=None, gaussian_std=None, 
                         uniform_lower=None, uniform_upper=None, anomaly_type=0):
        """
        Adds global noise to the dataset.

        Args:
            pct_data (float): Percentage of data points to modify.
            pct_features (float): Percentage of features to modify in each selected data point.
            noise_types (list): Types of noise to add ('gaussian', 'uniform', 'swap').
            gaussian_mean (float): Mean for Gaussian noise.
            gaussian_std (float): Standard deviation for Gaussian noise.
            uniform_lower (float): Lower bound for uniform noise.
            uniform_upper (float): Upper bound for uniform noise.
            anomaly_type (int): Type of anomaly to generate (0: easy, 1: medium, 2: interesting).

        Returns:
            tuple: Modified data and a dictionary of modifications made.
        """
        lof_model, if_model, svm_model = self.fit_validators()
        num_samples = int(pct_data * len(self.data))
        samples_to_modify = np.random.choice(np.arange(len(self.data)), num_samples, replace=False)
        modifications = {}

        for sample in samples_to_modify:
            loop_condition = True        
            easy_anomalies = np.empty((0, self.data.shape[1]))
            medium_anomalies = np.empty((0, self.data.shape[1]))
            interesting_anomalies = np.empty((0, self.data.shape[1]))
            
            while loop_condition:
                features_to_modify = np.random.choice(np.arange(self.data.shape[1]), 
                                                    int(pct_features * (self.data.shape[1])), replace=False)
                
                noisy_datapoint = self.data[sample].copy()
                
                if sample not in modifications:
                    modifications[sample] = {}  # initialize if not already

                for feature in features_to_modify:
                    if feature not in modifications[sample]:
                        modifications[sample][feature] = None  # initialize feature entry
                    picked_noise = np.random.choice(noise_types)  # a single perturbation for each feature for each datapoint

                    if picked_noise == 'gaussian' and modifications[sample][feature] is None:
                        mean = gaussian_mean if gaussian_mean is not None else 0
                        std = gaussian_std if gaussian_std is not None else np.std(self.data[:, feature])
                        noisy_datapoint[feature] = self.gaussian_noise(noisy_datapoint[feature], mean, std)
                        modifications[sample][feature] = 'gaussian'

                    elif picked_noise == 'uniform' and modifications[sample][feature] is None:
                        noisy_datapoint[feature] = self.uniform_noise(noisy_datapoint[feature], uniform_lower, uniform_upper)
                        modifications[sample][feature] = 'uniform'
                    
                    elif picked_noise == 'swap' and modifications[sample][feature] is None:
                        other_feature = np.random.choice(features_to_modify)
                        if (pct_features * self.data.shape[1]) >= 2:
                            while other_feature == feature:
                                other_feature = np.random.choice(features_to_modify)
                        
                        noisy_datapoint[feature], noisy_datapoint[other_feature] = noisy_datapoint[other_feature].copy(), noisy_datapoint[feature].copy() 

                        modifications[sample][feature] = 'swap'
                        modifications[sample][other_feature] = 'swap'

                easy_anomalies, medium_anomalies, interesting_anomalies = self.validated_samples(lof_model, if_model, svm_model,
                                                                                            np.array([noisy_datapoint]))
                
                if easy_anomalies.size > 0 and anomaly_type == 0:
                    self.data[sample] = easy_anomalies
                    loop_condition = False
                
                elif medium_anomalies.size > 0 and (anomaly_type == 0 or anomaly_type == 1):
                    self.data[sample] = medium_anomalies
                    loop_condition = False
                        
                elif interesting_anomalies.size > 0 and (anomaly_type == 0 or anomaly_type == 1 or anomaly_type == 2):
                    self.data[sample] = interesting_anomalies
                    loop_condition = False
                        
                else:
                    loop_condition = True
                    modifications[sample] = {}

        # add column that indicates the type of anomaly
        self.data = np.c_[self.data, np.zeros(self.data.shape[0])]
        self.data[samples_to_modify, -1] = 1
                    
        return modifications, self.data

    def generate_noise(self, *args, **kwargs):
        return self.generate_point_anomaly(*args, **kwargs)

    def add_noise(self, *args, **kwargs):
        return self.add_global_noise(*args, **kwargs)
    
    
    

class CollectiveAnomalyGenerator(AnomalyGenerator):
    def __init__(self, data):
        super().__init__(data)

    def generate_collective_noise(self, global_outlier, k=5, num_collective_anomalies=3, adaptive_cov=True, dispersion=0.8):
        """
        Generates collective noise (a cluster of anomalies) around a given global outlier.

        Args:
            global_outlier (np.ndarray): The global outlier data point.
            k (int): The number of nearest neighbors to consider for density estimation.
            num_collective_anomalies (int): The number of collective anomalies to generate.
            adaptive_cov (bool): Whether to use adaptive covariance.
            dispersion (float): Dispersion factor for covariance calculation.

        Returns:
            np.ndarray: Array containing the global outlier and generated collective anomalies.
        """
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors([global_outlier])
        
        knn_indices = indices[0][1:]
        knn = self.data[knn_indices]

        if adaptive_cov:
            distance_outlier = np.mean(distances[0][1:])
            distance_knn = np.mean(nbrs.kneighbors(knn)[0][:, 1:])
            distance_ratio = distance_knn / distance_outlier
            cov_scaling_factor = 1 - distance_ratio
            covariance = np.cov(knn.T) * dispersion * cov_scaling_factor
        else:
            covariance = np.cov(knn.T) * dispersion

        collective_anomalies = []
        while len(collective_anomalies) < (num_collective_anomalies - 1):
            candidate = np.random.multivariate_normal(global_outlier, covariance)
            dist_to_closest_inlier = nbrs.kneighbors([candidate], n_neighbors=2)[0][0][1]
            
            if np.linalg.norm(candidate - global_outlier) < dist_to_closest_inlier:
                collective_anomalies.append(candidate)

        return np.vstack([global_outlier, collective_anomalies])

    def add_collective_noise(self, global_noise_means, global_noise_sds, num_clusters=1, cluster_sizes=None):
        """
        Adds collective noise to the dataset with variable noise levels per cluster.

        Args:
            global_noise_means (list): Means for generating global point anomalies for each cluster.
            global_noise_sds (list): Standard deviations for generating global point anomalies for each cluster.
            num_clusters (int): The number of collective noise clusters to generate.
            cluster_sizes (list): Number of data points in each cluster.

        Returns:
            np.ndarray: The dataset with added collective noise.
        """
        if len(global_noise_means) != num_clusters or len(global_noise_sds) != num_clusters:
            raise ValueError("The number of means and standard deviations should match the number of clusters.")

        if cluster_sizes is None:
            cluster_sizes = [len(self.data) // num_clusters] * num_clusters

        for i, cluster_size in enumerate(cluster_sizes):
            global_anomaly = self.generate_point_anomaly(n_anomalies=1, mean=global_noise_means[i], std=global_noise_sds[i])
            collective_anomalies = self.generate_collective_noise(global_outlier=global_anomaly, num_collective_anomalies=cluster_size)
            self.data = np.vstack([self.data, collective_anomalies])

        # Add a column to indicate anomalies
        self.data = np.column_stack((self.data, np.zeros(self.data.shape[0])))
        self.data[-(sum(cluster_sizes)):, -1] = 3

        return self.data

    def generate_noise(self, *args, **kwargs):
        return self.generate_collective_noise(*args, **kwargs)

    def add_noise(self, *args, **kwargs):
        return self.add_collective_noise(*args, **kwargs)
    


class NoisyDatasetCreator:
    
    def __init__(self, clean_data):
        self.clean_data = clean_data
        self.global_generator = GlobalAnomalyGenerator(clean_data)
        self.collective_generator = CollectiveAnomalyGenerator(clean_data)

    
    def create_noisy_dataset(self, noise_types, global_noise_types, global_noise_parameters, sample_pct, feature_pct,
                             anomaly_means=None, anomaly_sds=None, clusters_proportions=None, anomaly_level=0,
                             save_path=".", dataset_name="dataset"):
        """
        Creates a noisy dataset with specified types and proportions of noise.

        Args:
            noise_types (list): Proportions of global, hidden, and collective noise.
            global_noise_types (list): Types of global noise to add.
            global_noise_parameters (list): Parameters for global noise.
            sample_pct (float): Percentage of samples to modify.
            feature_pct (float): Percentage of features to modify in global noise.
            anomaly_means (list): Means for collective anomalies.
            anomaly_sds (list): Standard deviations for collective anomalies.
            clusters_proportions (list): Proportions of collective anomaly clusters.
            anomaly_level (int): Level of anomaly to generate (0: easy, 1: medium, 2: interesting).
            save_path (str): Path to save the generated dataset.
            dataset_name (str): Name of the dataset.

        Returns:
            np.ndarray: The generated noisy dataset.
        """
        modified_data = self.clean_data.copy()
        
        num_glob = round(noise_types[0] * sample_pct * modified_data.shape[0])
        num_hidd = round(noise_types[1] * sample_pct * modified_data.shape[0])
        num_coll = round(noise_types[2] * sample_pct * modified_data.shape[0])
        
        
        if clusters_proportions is not None:
            cluster_sizes = [round(num_coll * prop) for prop in clusters_proportions]
            if any(size <= 1 for size in cluster_sizes):
                raise ValueError("All cluster sizes for the collective noise must be greater than 1")

        if noise_types[0] > 0:
            _, modified_data = self.global_generator.add_global_noise(
                num_glob / modified_data.shape[0], feature_pct, global_noise_types, 
                global_noise_parameters[0], global_noise_parameters[1], 
                global_noise_parameters[2], global_noise_parameters[3], 
                anomaly_type=anomaly_level
            )

        if noise_types[1] > 0:
            generator = BisectHOGen(modified_data[:, :modified_data.shape[1]-1], outlier_detection_method=pyod.models.ocsvm.OCSVM)
            hidden_anomalies = generator.fit_generate(gen_points=num_hidd)        
            hidden_anomalies = np.column_stack((hidden_anomalies, np.full(hidden_anomalies.shape[0], 2)))
            modified_data = np.vstack([modified_data, hidden_anomalies])
 
        if noise_types[2] > 0:
            modified_data = self.collective_generator.add_collective_noise(
                global_noise_means=anomaly_means,
                global_noise_sds=anomaly_sds,
                num_clusters=len(clusters_proportions),
                cluster_sizes=cluster_sizes
            )

        # Remove normal samples to maintain the original dataset size
        non_anomaly_indices = np.where(modified_data[:, -1] == 0)[0]
        indices_to_remove = np.random.choice(non_anomaly_indices, num_coll + num_hidd, replace=False)
        modified_data = np.delete(modified_data, indices_to_remove, axis=0)
        np.random.shuffle(modified_data)

        # Save the dataset
        os.makedirs(save_path, exist_ok=True)
        filename_parts = [dataset_name, "cont", str(sample_pct), "level", str(anomaly_level)]
        if noise_types[0] > 0:
            filename_parts.extend(["g", str(noise_types[0])])
        if noise_types[1] > 0:
            filename_parts.extend(["h", str(noise_types[1])])
        if noise_types[2] > 0:
            filename_parts.extend(["c", str(noise_types[2])])
        
        filename = f"{'_'.join(filename_parts)}.csv"
        file_path = os.path.join(save_path, filename)
        pd.DataFrame(modified_data).to_csv(file_path, index=False)
        print(f'Saved {file_path}')

        return modified_data
    

    def multiple_datasets_synthesis(self, proportions_list, sample_pcts, feature_pct, global_noise_types, 
                                    global_noise_parameters, anomaly_means=None, anomaly_sds=None, 
                                    clusters_proportions=None, anomaly_level=0, save_path=".", dataset_name="dataset"):
        """
        Generates multiple noisy datasets with different noise configurations.

        Args:
            proportions_list (list): List of noise type proportions to use.
            sample_pcts (list): List of sample percentages to use.
            feature_pct (float): Percentage of features to modify in global noise.
            global_noise_types (list): Types of global noise to add.
            global_noise_parameters (list): Parameters for global noise.
            anomaly_means (list): Means for collective anomalies.
            anomaly_sds (list): Standard deviations for collective anomalies.
            clusters_proportions (list): Proportions of collective anomaly clusters.
            anomaly_level (int): Level of anomaly to generate (0: easy, 1: medium, 2: interesting).
            save_path (str): Path to save the generated datasets.
            dataset_name (str): Base name for the datasets.
        """
        for sample_pct in sample_pcts:
            for proportions in proportions_list:
                print(proportions)
                if proportions[2] > 0.5 and (sample_pct * self.clean_data.shape[0] * proportions[2] >= 6):
                    # Two collective clusters
                    self.create_noisy_dataset(proportions, global_noise_types, global_noise_parameters, 
                                              sample_pct, feature_pct, anomaly_means, anomaly_sds, 
                                              clusters_proportions, anomaly_level, save_path, dataset_name)
                    
                    # One collective cluster
                    _ = self.create_noisy_dataset(proportions, global_noise_types, global_noise_parameters, 
                                              sample_pct, feature_pct, [anomaly_means[0]], [anomaly_sds[0]], 
                                              [1], anomaly_level, save_path, dataset_name)

                elif proportions[2] > 0 and (sample_pct * self.clean_data.shape[0] * proportions[2] < 3):
                    print("Skipping this proportion")
                    
                elif proportions[2] == 0:
                    _ = self.create_noisy_dataset(proportions, global_noise_types, global_noise_parameters,
                                              sample_pct, feature_pct, anomaly_level=anomaly_level,
                                              save_path=save_path, dataset_name=dataset_name)
                else:
                    _ = self.create_noisy_dataset(proportions, global_noise_types, global_noise_parameters,
                                              sample_pct, feature_pct, [anomaly_means[0]], [anomaly_sds[0]],
                                              [1], anomaly_level, save_path, dataset_name)


