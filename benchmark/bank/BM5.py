#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *


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

df['poutcome'] = df['poutcome'].map({'failure': -1,'nonexistent': 0,'success': 1})
df['default'] = df['default'].map({'yes': -1,'unknown': 0,'no': 1})
df['housing'] = df['housing'].map({'yes': -1,'unknown': 0,'no': 1})
df['loan'] = df['loan'].map({'yes': -1,'unknown': 0,'no': 1})

nominal = ['job','marital','education','contact','month','day_of_week']
df = pd.get_dummies(df, columns=nominal)


# In[3]:


pro_att_name = ['age'] # ['race', 'sex']
priv_class = [1] # ['White', 'Male']
reamining_cat_feat = []
seed = randrange(100)

y1_data_orig, y1_X, y1_y = load_bank_data(df, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_train, y1_data_orig_test = y1_data_orig.split([0.7], shuffle=True, seed=seed)

y1_X_train = y1_data_orig_train.features
y1_y_train = y1_data_orig_train.labels.ravel()
y1_X_test = y1_data_orig_test.features
y1_y_test = y1_data_orig_test.labels.ravel()

sc = StandardScaler()
y1_X_train = sc.fit_transform(y1_X_train)
y1_X_test = sc.fit_transform(y1_X_test)
y1_data_orig_train.features = y1_X_train
y1_data_orig_test.features = y1_X_test

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1], 'C': [1], 'probability': [True]}]
# Best parameter: 'C': 1, 'gamma': 0.1, 'kernel': 'rbf'

#SVC = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='precision')
y1_svc = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
y1_mdl = y1_svc.fit(y1_X_train, y1_y_train)
plot_model_performance(y1_mdl, y1_X_test, y1_y_test)

