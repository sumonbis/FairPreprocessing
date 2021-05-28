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
# from aif360.algorithms.preprocessing import LFR, Reweighing, DisparateImpactRemover
# from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
# from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification
#from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas

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
