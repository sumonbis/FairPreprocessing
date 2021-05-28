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
df = df.drop(columns = ['pdays'])

# Create a one-hot encoding of the categorical variables.
cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


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

import xgboost
y1_xgb = xgboost.XGBClassifier(n_estimators=100, 
                              learning_rate=0.08, 
                              gamma=0, 
                              subsample=0.75,
                              colsample_bytree=1, 
                              max_depth=7)

y1_mdl = y1_xgb.fit(y1_X_train, y1_y_train)
plot_model_performance(y1_mdl, y1_X_test, y1_y_test)

