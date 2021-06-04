#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/adult-'

d_fields = ['Classifier', 'Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


train_path = '../../data/adult/adult.data'
test_path = '../../data/adult/adult.test'
column_names = ['age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'income-per-year']
na_values=['?']
train = pd.read_csv(train_path, header=None, names=column_names, skipinitialspace=True, na_values=na_values)
test = pd.read_csv(test_path, header=0, names=column_names, skipinitialspace=True, na_values=na_values)
df = pd.concat([test, train], ignore_index=True)

##### Drop na values
dropped = df.dropna()
count = df.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df = dropped

## Feature selection
# features_to_keep = ['workclass', 'education-num', 'marital-status', 'race', 'sex', 'relationship', 'capital-gain', 'capital-loss', 'income-per-year']
# # cat_feat = ['sex', 'workclass', 'marital-status', 'relationship']
# df = df[features_to_keep]

# Create a one-hot encoding of the categorical variables.
cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

# for feature in cat_feat:
#     le = LabelEncoder()
#     df[feature] = le.fit_transform(df[feature])


# In[3]:


seed = 42 # randrange(100)
y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['race']
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) # 


# In[4]:


### Imputation

# #### Drop na values
# y1_train = y1_train.dropna()
# y1_test = y1_test.dropna()
# y2_test = y1_test.dropna()

# le = LabelEncoder()
# for feature in cat_feat:
#     y1_train[feature] = le.fit_transform(y1_train[feature])
#     y2_test[feature] = le.fit_transform(y2_test[feature])
#     y1_test[feature] = le.fit_transform(y1_test[feature])

# from sklearn.impute import KNNImputer, MissingIndicator
# # imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
# imputer = SimpleImputer(missing_values=np.NaN, strategy='constant', fill_value='unknown')
# # imputer = IterativeImputer(missing_values=np.NaN, initial_strategy='most_frequent')
# # imputer = KNNImputer(missing_values=np.NaN)
# # imputer = MissingIndicator(missing_values=np.NaN, strategy='most_frequent')
# y2_train_imputed = pd.DataFrame(imputer.fit_transform(y2_train))
# y2_train_imputed.columns = y2_train.columns
# y2_train_imputed.index = y2_train.index
# y2_train = y2_train_imputed

# for feature in cat_feat:
#     y2_train[feature] = le.fit_transform(y2_train[feature])


# In[5]:


pro_att_name = ['race'] # ['race', 'sex']
priv_class = ['White'] # ['White', 'Male']
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_adult_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_adult_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)

y1_data_orig_train, y1_X_train, y1_y_train = load_adult_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_adult_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)


# In[6]:


# from imblearn.pipeline import Pipeline as Pipe
# from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, AllKNN
# from imblearn.combine import SMOTEENN

# over = SVMSMOTE(sampling_strategy='auto')
# under = AllKNN(sampling_strategy='auto')
# combine = SMOTEENN(sampling_strategy='auto')
# step_over = [('o', over)] # , ('u', under)
# step_under = [('u', under)] # , ('u', under)
# step_combined = [('c', combine)] # , ('u', under)
# # p = Pipe(steps=under)
# y2_X_train, y2_y_train = combine.fit_resample(y2_X_train, y2_y_train)


# In[7]:


# sc = StandardScaler()
# sc = MinMaxScaler()
# sc = MaxAbsScaler()
# sc = RobustScaler()
sc = Normalizer(norm='l1')

from sklearn.preprocessing import QuantileTransformer, PowerTransformer
# sc = QuantileTransformer()
# sc = PowerTransformer()

y2_X_train = sc.fit_transform(y2_X_train)
y2_X_test = sc.fit_transform(y2_X_test)
y2_data_orig_train.features = y2_X_train
y2_data_orig_test.features = y2_X_test


# In[8]:


from sklearn.feature_selection import SelectFpr, SelectPercentile, VarianceThreshold
# pca = PCA(n_components=5)
# pca = SparsePCA(n_components=5)
# pca = KernelPCA(n_components=5)

# pca = SelectKBest(k=5)
# pca = SelectFpr()
# pca = SelectPercentile()

# trained = pca.fit(y2_X_train, y2_y_train)
# y2_X_train = trained.transform(y2_X_train)
# y2_X_test = trained.transform(y2_X_test)
# y2_data_orig_test.features = y2_X_test


# In[9]:


from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier

y2_lr = LogisticRegression()
y1_lr = LogisticRegression()

y2_gbc = GradientBoostingClassifier()
y1_gbc = GradientBoostingClassifier()

y2_cbc = CatBoostClassifier()
y1_cbc = CatBoostClassifier()

y2_xgb = XGBClassifier()
y1_xgb = XGBClassifier()

y2_dct = DecisionTreeClassifier()
y1_dct = DecisionTreeClassifier()

classifiers = [(y1_lr, y2_lr), (y1_gbc, y2_gbc), (y1_cbc, y2_cbc), (y1_xgb, y2_xgb), (y1_dct, y2_dct)]


# In[10]:


clfs = ['DCT', 'RFT', 'XGB', 'SVC', 'KNC']
i = 0
for clf in classifiers:
    y1_clf = clf[0]
    y2_clf = clf[1]
    
    y2_mdl = y2_clf.fit(y2_X_train, y2_y_train)
    y1_mdl = y1_clf.fit(y1_X_train, y1_y_train)

    # plot_model_performance(y2_mdl, y2_X_test, y2_y_test)
    y1_pred, y1_fair = get_fair_metrics_and_plot('filename', y1_data_orig_test, y1_mdl)
    y2_pred, y2_fair = get_fair_metrics_and_plot('filename', y2_data_orig_test, y2_mdl)

    y1_fair = y1_fair.drop(['DI', 'CNT', 'TI'], axis=1)
    y2_fair = y2_fair.drop(['DI', 'CNT', 'TI'], axis=1)
    
    CVR, CVD, AVR_EOD, AVD_EOD, AVR_SPD, AVD_SPD, AVD_AOD, AV_ERD = compute_new_metrics(y1_data_orig_test, y1_pred, y2_pred)
    row_y1 = y1_fair.iloc[[0]].values[0].tolist()
    row_y2 = y2_fair.iloc[[0]].values[0].tolist()
    diff = []
    
    # diff.append(CVR)
    # diff.append(CVD)
    diff.append(AVD_SPD)
    diff.append(AVD_EOD)
    diff.append(AVD_AOD)
    diff.append(AV_ERD)

    stage = 'SMOTE'
    model_name = 'german'
    # diff = diff_df.iloc[0].values.tolist()
    diff.insert(0, stage)
    diff.insert(0, model_name)
    diff.insert(0, clfs[i])

    cols = ['Classifier', 'Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD']
    # metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
    diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(diff)
    i += 1


# In[ ]:




