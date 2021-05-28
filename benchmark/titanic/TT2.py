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
## Feature Engineering
df['Family'] =  df["Parch"] + df["SibSp"]
df['Family'].loc[df['Family'] > 0] = 1
df['Family'].loc[df['Family'] == 0] = 0
df = df.drop(['SibSp','Parch'], axis=1)
df = df.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)

y1_df = y1_df.drop(['PassengerId','Name'], axis=1)


# In[4]:

# Missing value
average_age_titanic   = y1_df["Age"].mean()
std_age_titanic       = y1_df["Age"].std()
count_nan_age_titanic = y1_df["Age"].isnull().sum()
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
y1_df["Age"][np.isnan(y1_df["Age"])] = rand_1

y1_df["Embarked"] = y1_df["Embarked"].fillna("S")
y1_df["Fare"].fillna(y1_df["Fare"].median(), inplace=True)

y1_df['Fare'] = y1_df['Fare'].astype(int)
y1_df[ 'Cabin' ] = y1_df.Cabin.fillna( 'U' )
y1_df[ 'Ticket' ] = y1_df.Ticket.fillna( 'X' )
y1_df = y1_df.dropna()
# One-hot encoder
cat_feat = ['Embarked', 'Ticket', 'Cabin']
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')


average_age_titanic   = df["Age"].mean()
std_age_titanic       = df["Age"].std()
count_nan_age_titanic = df["Age"].isnull().sum()
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
df["Age"][np.isnan(df["Age"])] = rand_1

df["Embarked"] = df["Embarked"].fillna("S")
df["Fare"].fillna(df["Fare"].median(), inplace=True)

df['Fare'] = df['Fare'].astype(int)
df = df.dropna()
# One-hot encoder
cat_feat = ['Embarked']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


# In[5]:


seed = randrange(100)
y1_train, y1_test = train_test_split(y1_df, test_size = 0.3, random_state = seed) #

pro_att_name = ['Sex']
priv_class = [1]
reamining_cat_feat = []

y1_data_orig_train, y1_X_train, y1_y_train = load_titanic_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_titanic_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)

y1_xgb = RandomForestClassifier(n_estimators=100)
y1_mdl = y1_xgb.fit(y1_X_train, y1_y_train)
plot_model_performance(y1_mdl, y1_X_test, y1_y_test)
