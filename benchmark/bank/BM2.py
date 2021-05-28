#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *


# In[2]:


def custom_get_fair_metrics_and_plot(data, model, model_aif=False):
    pred = model.predict(data).labels if model_aif else model.predict(data.features)
    pred = (pred >= 0.5) * 1
    fair = fair_metrics(fname, data, pred)
    return (pred, fair)


# In[3]:


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

## Feature selection
# features_to_keep = []
# df = df[features_to_keep]
y2_df = df.copy()
# Create a one-hot encoding of the categorical variables.
cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

for feature in cat_feat:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])


# In[4]:


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

import lightgbm as lgb
from xgboost.sklearn import XGBClassifier

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

y1_lgb_train = lgb.Dataset(data=y1_X_train, label=y1_y_train,  free_raw_data=False)
y1_lgb_eval = lgb.Dataset(data=y1_X_test, label=y1_y_test, reference=y1_lgb_train,  free_raw_data=False)
y1_evals_result={}

y1_md = XGBClassifier()
y1_mdl = lgb.train(params,
                y1_lgb_train,
                valid_sets = y1_lgb_eval,
                num_boost_round= 150,
                early_stopping_rounds= 25,
                evals_result=y1_evals_result)

pred = y1_mdl.predict(y1_data_orig_test.features)
pred = (pred >= 0.5) * 1


# In[ ]:
