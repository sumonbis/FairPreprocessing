#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *


# In[2]:


# Load data
train = pd.read_csv('../../data/titanic/train.csv')
test = pd.read_csv('../../data/titanic/test.csv')
df = train


# In[3]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

y1_df = df.copy()

df['Child'] = df['Age']<=10
df['Cabin_known'] = df['Cabin'].isnull() == False
df['Age_known'] = df['Age'].isnull() == False
df['Family'] = df['SibSp'] + df['Parch']
df['Alone']  = (df['SibSp'] + df['Parch']) == 0
df['Large_Family'] = (df['SibSp']>2) | (df['Parch']>3)
df['Deck'] = df['Cabin'].str[0]
df['Deck'] = df['Deck'].fillna(value='U')
df['Ttype'] = df['Ticket'].str[0]
df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
df['Fare_cat'] = pd.DataFrame(np.floor(np.log10(df['Fare'] + 1))).astype('int')
df['Bad_ticket'] = df['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])
df['Young'] = (df['Age']<=30) | (df['Title'].isin(['Master','Miss','Mlle']))
df['Shared_ticket'] = np.where(df.groupby('Ticket')['Name'].transform('count') > 1, 1, 0)
df['Ticket_group'] = df.groupby('Ticket')['Name'].transform('count')
df['Fare_eff'] = df['Fare']/df['Ticket_group']
df['Fare_eff_cat'] = np.where(df['Fare_eff']>16.0, 2, 1)
df['Fare_eff_cat'] = np.where(df['Fare_eff']<8.5,0,df['Fare_eff_cat'])
df['Age'].fillna(df['Age'].median(), inplace = True)
df = df.fillna({"Embarked": "S"})

cat_feat = ['Deck', 'Title', 'Ttype',]
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

# drop_column = ['Name', 'Ticket', 'Fare', 'Cabin']
# df.drop(drop_column, axis=1, inplace = True)

df['Age'].fillna(df['Age'].median(), inplace = True)
df = df.fillna({"Embarked": "S"})
df['Cabin'].fillna(df['Cabin'].mode(), inplace = True)

# One-hot encoder
cat_feat = ['Cabin', 'Ticket', 'Embarked']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

df = df.drop(['PassengerId'], axis = 1)
df = df.drop(['Name'], axis = 1)


# In[4]:


seed = randrange(100)
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) # 

pro_att_name = ['Sex']
priv_class = [1]
reamining_cat_feat = []

y1_data_orig_train, y1_X_train, y1_y_train = load_titanic_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_titanic_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)

import xgboost as xgb

y1_xgb = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scale_pos_weight=1)

y1_mdl = y1_xgb.fit(y1_X_train, y1_y_train)

plot_model_performance(y1_mdl, y1_X_test, y1_y_test)

