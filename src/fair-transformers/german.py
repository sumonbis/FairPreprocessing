#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/german-'

d_fields = ['Classifier', 'Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


filepath = '../../data/german/german.data'
column_names = ['status', 'month', 'credit_history',
            'purpose', 'credit_amount', 'savings', 'employment',
            'investment_as_income_percentage', 'personal_status',
            'other_debtors', 'residence_since', 'property', 'age',
            'installment_plans', 'housing', 'number_of_credits',
            'skill_level', 'people_liable_for', 'telephone',
            'foreign_worker', 'credit']
na_values=[]
df = pd.read_csv(filepath, sep=' ', header=None, names=column_names,na_values=na_values)
# df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
df = german_custom_preprocessing(df)
feat_to_drop = ['personal_status']
df = df.drop(feat_to_drop, axis=1)

cat_feat = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


# In[3]:


seed = 42 # randrange(100)
y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['loan']
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) # 


# In[4]:


pro_att_name = ['sex'] # ['sex', 'age']
priv_class = [1]
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_german_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_german_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)

y1_data_orig_train, y1_X_train, y1_y_train = load_german_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_german_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)

y1_test_df = y1_data_orig_test.copy()


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

y2_dct = DecisionTreeClassifier()
y1_dct = DecisionTreeClassifier()

y2_rf = RandomForestClassifier()
y1_rf = RandomForestClassifier()

y2_xgb = XGBClassifier()
y1_xgb = XGBClassifier()

y2_svc = SVC(random_state=42)
y1_svc = SVC(random_state=42)

y2_gbc = KNeighborsClassifier(n_neighbors=10)
y1_gbc = KNeighborsClassifier(n_neighbors=10)


classifiers = [(y1_dct, y2_dct), (y1_rf, y2_rf), (y1_xgb, y2_xgb), (y1_svc, y2_svc), (y1_gbc, y2_gbc)]


# In[9]:


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

