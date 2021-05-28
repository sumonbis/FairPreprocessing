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

### Missing value
df['Age'].fillna(df['Age'].median(), inplace = True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)
df['Fare'].fillna(df['Fare'].median(), inplace = True)
df = df.dropna()

### Feature engineering
## Drop feature

drop_column = ['PassengerId', 'Cabin', 'Ticket', 'Name', 'Parch']
## generation
df['FamilySize'] = df ['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 1
df['IsAlone'].loc[df['FamilySize'] > 1] = 0
df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (df['Title'].value_counts() < stat_min) #this will create a true false series with title name as index
df['Title'] = df['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

df.drop(drop_column, axis=1, inplace = True)

### Encoding
y1_df = df.copy()
## Binning
df['Fare'] = pd.qcut(df['Fare'], 4)
df['Age'] = pd.cut(df['Age'].astype(int), 5)

## LabelEncoder
label = LabelEncoder()
df['Embarked'] = label.fit_transform(df['Embarked'])
df['Title'] = label.fit_transform(df['Title'])
df['Age'] = label.fit_transform(df['Age'])
df['Fare'] = label.fit_transform(df['Fare'])

## One-hot encoder
cat_feat = ['Title', 'Embarked']
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')


# In[4]:


seed = randrange(100)
y1_train, y1_test = train_test_split(y1_df, test_size = 0.3, random_state = seed) # 

pro_att_name = ['Sex']
priv_class = [1]
reamining_cat_feat = []
y1_data_orig_train, y1_X_train, y1_y_train = load_titanic_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_titanic_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)

from xgboost import XGBClassifier
y1_xgb = XGBClassifier(learning_rate= 0.01, max_depth= 4, n_estimators= 300, seed= 0)
y1_mdl = y1_xgb.fit(y1_X_train, y1_y_train)

plot_model_performance(y1_mdl, y1_X_test, y1_y_test)

