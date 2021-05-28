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

#### Drop na values
dropped = df.dropna()
count = df.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df = dropped

df['age'] = df['age'].apply(lambda x: np.float(x >= 25))

# Create a one-hot encoding of the categorical variables.
cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

labelencoder_X = LabelEncoder()
df['job']      = labelencoder_X.fit_transform(df['job'])
df['marital']  = labelencoder_X.fit_transform(df['marital'])
df['education']= labelencoder_X.fit_transform(df['education'])
df['default']  = labelencoder_X.fit_transform(df['default'])
df['housing']  = labelencoder_X.fit_transform(df['housing'])
df['loan']     = labelencoder_X.fit_transform(df['loan'])
df['contact']     = labelencoder_X.fit_transform(df['contact'])
df['month']       = labelencoder_X.fit_transform(df['month'])
df['day_of_week'] = labelencoder_X.fit_transform(df['day_of_week'])
df['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)

def duration(data):
    data.loc[data['duration'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']    = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration'] = 4
    data.loc[data['duration']  > 644.5, 'duration'] = 5

    return data

df = duration(df)


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

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sc = StandardScaler()
y1_X_train = sc.fit_transform(y1_X_train)
y1_X_test = sc.transform(y1_X_test)
y1_data_orig_train.features = y1_X_train
y1_data_orig_test.features = y1_X_test

y1_xgb = XGBClassifier()
y1_mdl = y1_xgb.fit(y1_X_train, y1_y_train)

plot_model_performance(y1_mdl, y1_X_test, y1_y_test)
