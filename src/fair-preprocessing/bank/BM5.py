#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/bank5-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


file_path = '../../../data/bank/bank-additional-full.csv'

column_names = []
na_values=['unknown']

df = pd.read_csv(file_path, sep=';', na_values=na_values)

### Drop na values
dropped = df.dropna()
count = df.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df = dropped

df['age'] = df['age'].apply(lambda x: np.float(x >= 25))

# y1_df = df.copy()

# df['poutcome'] = df['poutcome'].map({'failure': -1,'nonexistent': 0,'success': 1})
# df['default'] = df['default'].map({'yes': -1,'unknown': 0,'no': 1})
# df['housing'] = df['housing'].map({'yes': -1,'unknown': 0,'no': 1})
# df['loan'] = df['loan'].map({'yes': -1,'unknown': 0,'no': 1})

# nominal = ['job','marital','education','contact','month','day_of_week']
# df = pd.get_dummies(df, columns=nominal)

y1_nominal = ['job','marital','education','contact','month','day_of_week', 'poutcome', 'default', 'housing', 'loan']
df = pd.get_dummies(df, columns=y1_nominal)


# In[3]:


# y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = 0) #, stratify=df['race']
# y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = 0)


# In[ ]:



pro_att_name = ['age'] # ['race', 'sex']
priv_class = [1] # ['White', 'Male']
reamining_cat_feat = []
seed = randrange(100)

y2_data_orig, y2_X, y2_y = load_bank_data(df, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_train, y2_data_orig_test = y2_data_orig.split([0.7], shuffle=True, seed=seed)

y2_X_train = y2_data_orig_train.features
y2_y_train = y2_data_orig_train.labels.ravel()
y2_X_test = y2_data_orig_test.features
y2_y_test = y2_data_orig_test.labels.ravel()

y1_data_orig, y1_X, y1_y = load_bank_data(df, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_train, y1_data_orig_test = y1_data_orig.split([0.7], shuffle=True, seed=seed)

y1_X_train = y1_data_orig_train.features
y1_y_train = y1_data_orig_train.labels.ravel()
y1_X_test = y1_data_orig_test.features
y1_y_test = y1_data_orig_test.labels.ravel()




sc = StandardScaler()
y2_X_train = sc.fit_transform(y2_X_train)
y2_X_test = sc.fit_transform(y2_X_test)
y2_data_orig_train.features = y2_X_train
y2_data_orig_test.features = y2_X_test

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1], 'C': [1], 'probability': [True]}]
# Best parameter: 'C': 1, 'gamma': 0.1, 'kernel': 'rbf'

#SVC = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='precision')
y2_svc = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
y2_mdl = y2_svc.fit(y2_X_train, y2_y_train)

#SVC = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='precision')
y1_svc = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
y1_mdl = y1_svc.fit(y1_X_train, y1_y_train)



# plot_model_performance(y2_mdl, y2_X_test, y2_y_test)
y1_pred, y1_fair = get_fair_metrics_and_plot('filename', y1_data_orig_test, y1_mdl)
y2_pred, y2_fair = get_fair_metrics_and_plot('filename', y2_data_orig_test, y2_mdl)



y1_fair = y1_fair.drop(['DI', 'CNT', 'TI'], axis=1)
y2_fair = y2_fair.drop(['DI', 'CNT', 'TI'], axis=1)
CVR, CVD, AVR_EOD, AVD_EOD, AVR_SPD, AVD_SPD, AVD_AOD, AV_ERD = compute_new_metrics(y2_data_orig_test, y1_pred, y2_pred)
row_y1 = y1_fair.iloc[[0]].values[0].tolist()
row_y2 = y2_fair.iloc[[0]].values[0].tolist()
diff = []

# diff.append(CVR)
# diff.append(CVD)
diff.append(AVD_SPD)
diff.append(AVD_EOD)
diff.append(AVD_AOD)
diff.append(AV_ERD)

for i in range(len(row_y2)):
    if(i < 2):
        change = row_y2[i] - row_y1[i]
    else:
        break;
    diff.append(change)

stage = 'StandardScaler'
model_name = 'bank5'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)

