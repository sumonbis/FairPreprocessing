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

## Custom feature
df["Age"] = df["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)

df["CabinBool"] = (df["Cabin"].notnull().astype('int'))

df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
df['Title'] = df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
df['Title'] = df['Title'].map(title_mapping)
df['Title'] = df['Title'].fillna(0)

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
for x in range(len(train["AgeGroup"])):
    if df["AgeGroup"][x] == "Unknown":
        df["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
df['AgeGroup'] = df['AgeGroup'].map(age_mapping)

df = df.fillna({"Embarked": "S"})
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
df['Embarked'] = df['Embarked'].map(embarked_mapping)
df['FareBand'] = pd.qcut(df['Fare'], 4, labels = [1, 2, 3, 4])

df['Age'].fillna(df['Age'].median(), inplace = True)
df = df.fillna({"Embarked": "S"})
df['Embarked'] = df['Embarked'].map(embarked_mapping)
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

from sklearn.ensemble import GradientBoostingClassifier
y1_model = GradientBoostingClassifier()
y1_mdl = y1_model.fit(y1_X_train,y1_y_train) 

plot_model_performance(y1_mdl, y1_X_test, y1_y_test)

