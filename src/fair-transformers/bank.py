#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/bank-'

d_fields = ['Classifier', 'Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


file_path = '../../data/bank/bank-additional-full.csv'

column_names = []
na_values=['unknown']

df = pd.read_csv(file_path, sep=';', na_values=na_values)

### Drop na values
dropped = df.dropna()
count = df.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df = dropped

df['age'] = df['age'].apply(lambda x: np.float(x >= 25))

## Feature selection
# features_to_keep = []
# df = df[features_to_keep]

# Create a one-hot encoding of the categorical variables.
cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

# for feature in cat_feat:
#     le = LabelEncoder()
#     y2_df[feature] = le.fit_transform(y2_df[feature])

# y2_df = df.copy()
# num_cols = ['duration', 'campaign', 'pdays',
#        'previous', 'emp.var.rate', 'cons.price.idx',
#        'cons.conf.idx', 'euribor3m', 'nr.employed']
# for feature in num_cols:
#     kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
#     y2_df[feature] = kb.fit_transform(y2_df[[feature]])


# In[3]:


seed = 42 # randrange(100)
y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['loan']
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) # 


# In[4]:


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


pro_att_name = ['age'] # ['race', 'sex']
priv_class = [1] # ['White', 'Male']
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_bank_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_bank_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)

y1_data_orig_train, y1_X_train, y1_y_train = load_bank_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_bank_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)

y1_test_df = y1_data_orig_test.copy()


# In[6]:


from sklearn.feature_selection import SelectFpr, SelectPercentile, VarianceThreshold
# pca = PCA(n_components=5)
# pca = SparsePCA(n_components=5)
# pca = KernelPCA(n_components=5)

# pca = SelectKBest(k=5)
# pca = SelectFpr()
pca = SelectPercentile()

trained = pca.fit(y2_X_train, y2_y_train)
y2_X_train = trained.transform(y2_X_train)
y2_X_test = trained.transform(y2_X_test)
y2_data_orig_test.features = y2_X_test


# In[7]:


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


# In[8]:


# sc = StandardScaler()
# sc = MinMaxScaler()
# sc = MaxAbsScaler()
# sc = RobustScaler()
# sc = Normalizer(norm='l1')

# from sklearn.preprocessing import QuantileTransformer, PowerTransformer
# sc = QuantileTransformer()
# sc = PowerTransformer()
# y2_X_train = sc.fit_transform(y2_X_train)
# y2_X_test = sc.fit_transform(y2_X_test)
# y2_data_orig_train.features = y2_X_train
# y2_data_orig_test.features = y2_X_test


# In[9]:


from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import lightgbm as lgb

y2_lr = LogisticRegression()
y1_lr = LogisticRegression()

y2_rf = RandomForestClassifier()
y1_rf = RandomForestClassifier()

y2_gbc = GradientBoostingClassifier()
y1_gbc = GradientBoostingClassifier()

y2_xgb = XGBClassifier()
y1_xgb = XGBClassifier()

y2_svc = SVC()
y1_svc = SVC()

classifiers = [(y1_lr, y2_lr), (y1_rf, y2_rf), (y1_gbc, y2_gbc), (y1_xgb, y2_xgb), (y1_svc, y2_svc)]


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
    CVR, CVD, AVR_EOD, AVD_EOD, AVR_SPD, AVD_SPD, AVD_AOD, AV_ERD = compute_new_metrics(y1_test_df, y1_pred, y2_pred)
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

