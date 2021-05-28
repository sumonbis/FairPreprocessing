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


def name_converted(feature):
    result = ''
    if feature in ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col', 'Rev', 'Dona', 'Dr']:
        result = 'rare'
    elif feature in ['Ms', 'Mlle']:
        result = 'Miss'
    elif feature == 'Mme':
        result = 'Mrs'
    else:
        result = feature
    return result
def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a
def fare_group(fare):
    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a


# In[4]:


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


# In[5]:


seed = randrange(100)
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['loan']

pro_att_name = ['Sex']
priv_class = [1]
reamining_cat_feat = []

y1_data_orig_train, y1_X_train, y1_y_train = load_titanic_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_titanic_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)

from sklearn.ensemble import BaggingClassifier
y1_model = BaggingClassifier(base_estimator=None, bootstrap=True, bootstrap_features=False,
                  max_features=1.0, max_samples=1.0, n_estimators=180,
                  n_jobs=None, oob_score=False, random_state=None, verbose=0,
                  warm_start=False)
y1_mdl = y1_model.fit(y1_X_train, y1_y_train)

plot_model_performance(y1_mdl, y1_X_test, y1_y_test)

