import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from itertools import combinations
from pyod.models.ocsvm import OCSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from hog_bisect.bisect import BisectHOGen
import pyod
from pymfe.mfe import MFE
from anomaly_generation import *
from preprocessing import *


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

preproc = DataPreprocessing('/home/enosim/Desktop/PoAC for Noise Reduction/Noisy_Datasets_Synthesis/Datasets/iris.csv', numpy=True)
synthesizer = NoisyDatasetCreator(preproc.data)

synthesizer.multiple_datasets_synthesis(proportions_list=proportions_list, sample_pcts=[0.05, 0.1], feature_pct=0.5, 
                  global_noise_types=["gaussian", "swap"], global_noise_parameters=[1.2, 0.1, 1, 1], anomaly_means=[2, -2],
                  anomaly_sds=[0.1, 0.1], clusters_proportions=[0.5, 0.5], anomaly_level=0, save_path='./Noisy_Datasets',
                  dataset_name='iris' )
