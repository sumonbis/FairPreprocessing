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

cat_feat = ['job', 'default', 'housing', 'contact', 'month', 'day_of_week', 'poutcome']
df.drop(['marital', 'education'], axis=1, inplace=True)

# Create a one-hot encoding of the categorical variables.
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


# In[3]:


seed = randrange(100)
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed, stratify=df['loan']) # 

pro_att_name = ['age'] # ['race', 'sex']
priv_class = [1] # ['White', 'Male']
reamining_cat_feat = ['loan']

y1_data_orig_train, y1_X_train, y1_y_train = load_bank_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_bank_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)

sc = StandardScaler()
y1_X_train = sc.fit_transform(y1_X_train)
y1_X_test = sc.fit_transform(y1_X_test)
y1_data_orig_train.features = y1_X_train
y1_data_orig_test.features = y1_X_test

from sklearn.ensemble import GradientBoostingClassifier
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1], 'C': [1], 'probability': [True]}]
# Best parameter: 'C': 1, 'gamma': 0.1, 'kernel': 'rbf'

# Gradient Boosting Classifier
y1_gbc = GradientBoostingClassifier()
y1_mdl = y1_gbc.fit(y1_X_train, y1_y_train)
plot_model_performance(y1_mdl, y1_X_test, y1_y_test)

