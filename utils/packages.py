# numpy and pandas for data manipulation
from time import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os.path
from random import randrange
#import tensorflow as tf

# AIF360 Library
from aif360.datasets import *

# Scikit-learn Library
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, SVMSMOTE

from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, KBinsDiscretizer, Normalizer, MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.decomposition import PCA, NMF, SparsePCA, KernelPCA
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold
