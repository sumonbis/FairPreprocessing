#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.path.append('../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *


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
df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
df = german_custom_preprocessing(df)
feat_to_drop = ['personal_status']
df = df.drop(feat_to_drop, axis=1)

cat_feat = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
num_feat = ['residence_since', 'age', 'investment_as_income_percentage', 'credit_amount', 'number_of_credits', 'people_liable_for', 'month']


# In[3]:


#####  Pipeline  #####
from collections import defaultdict
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier


# In[4]:


seed = randrange(100)
y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['race']
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) #

pro_att_name = ['age'] # ['sex', 'age']
priv_class = [1]
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_german_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_german_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)

y1_data_orig_train, y1_X_train, y1_y_train = load_german_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_german_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)


# In[5]:


sc = StandardScaler()
trained = sc.fit(y1_X_train)
y1_X_train = trained.transform(y1_X_train)
y1_X_test = trained.transform(y1_X_test)

y1_data_orig_train.features = y1_X_train
y1_data_orig_test.features = y1_X_test


# In[6]:


# Balanced Dataset
sm = SMOTE(sampling_strategy='auto')
y1_X_train, y1_y_train = sm.fit_sample(y1_X_train, y1_y_train)


# In[7]:


eval_set_y1=[(y1_X_train, y1_y_train), (y1_X_test, y1_y_test)]
eval_set_y2=[(y2_X_train, y2_y_train), (y2_X_test, y2_y_test)]

params2={
    'n_estimators':3000,
    'objective': 'binary:logistic',
    'learning_rate': 0.005,
    #'gamma':0.01,
    'subsample':0.555,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'max_depth':8,
    #'seed':1024,
    'n_jobs' : -1
}

y1_model = XGBClassifier(**params2).fit(y1_X_train, y1_y_train, eval_set=eval_set_y1, early_stopping_rounds = 100, verbose=0)
y1_model.set_params(**{'n_estimators': y1_model.best_ntree_limit})
y1_mdl = y1_model.fit(y1_X_train, y1_y_train)


# In[8]:


plot_model_performance(y1_mdl, y1_X_test, y1_y_test)
