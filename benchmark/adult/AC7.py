#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *


# In[2]:


train_path = '../../data/adult/adult.data'
test_path = '../../data/adult/adult.test'

column_names = ['age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'income-per-year']
na_values=['?']

train = pd.read_csv(train_path, header=None, names=column_names, 
                    skipinitialspace=True, na_values=na_values)
test = pd.read_csv(test_path, header=0, names=column_names,
                   skipinitialspace=True, na_values=na_values)

df = pd.concat([test, train], ignore_index=True)

##### Drop na values
dropped = df.dropna()
count = df.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df = dropped


# In[3]:


df['education'].replace('Preschool', 'dropout',inplace=True)
df['education'].replace('10th', 'dropout',inplace=True)
df['education'].replace('11th', 'dropout',inplace=True)
df['education'].replace('12th', 'dropout',inplace=True)
df['education'].replace('1st-4th', 'dropout',inplace=True)
df['education'].replace('5th-6th', 'dropout',inplace=True)
df['education'].replace('7th-8th', 'dropout',inplace=True)
df['education'].replace('9th', 'dropout',inplace=True)
df['education'].replace('HS-Grad', 'HighGrad',inplace=True)
df['education'].replace('HS-grad', 'HighGrad',inplace=True)
df['education'].replace('Some-college', 'CommunityCollege',inplace=True)
df['education'].replace('Assoc-acdm', 'CommunityCollege',inplace=True)
df['education'].replace('Assoc-voc', 'CommunityCollege',inplace=True)
df['education'].replace('Bachelors', 'Bachelors',inplace=True)
df['education'].replace('Masters', 'Masters',inplace=True)
df['education'].replace('Prof-school', 'Masters',inplace=True)
df['education'].replace('Doctorate', 'Doctorate',inplace=True)

df['marital-status'].replace('Never-married', 'NotMarried',inplace=True)
df['marital-status'].replace(['Married-AF-spouse'], 'Married',inplace=True)
df['marital-status'].replace(['Married-civ-spouse'], 'Married',inplace=True)
df['marital-status'].replace(['Married-spouse-absent'], 'NotMarried',inplace=True)
df['marital-status'].replace(['Separated'], 'Separated',inplace=True)
df['marital-status'].replace(['Divorced'], 'Separated',inplace=True)
df['marital-status'].replace(['Widowed'], 'Widowed',inplace=True)

df['age'] = pd.cut(df['age'], 20)
df['hours-per-week'] = pd.cut(df['hours-per-week'], 10)


# In[4]:


# Create a one-hot encoding of the categorical variables.
cat_feat = ['age', 'hours-per-week', 'sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


# In[5]:


seed = randrange(100)
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) # 

pro_att_name = ['race'] # ['race', 'sex']
priv_class = ['White'] # ['White', 'Male']
reamining_cat_feat = []

y1_data_orig_train, y1_X_train, y1_y_train = load_adult_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_adult_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)
y1_test_org = y1_data_orig_test.copy()


# In[6]:


sc = StandardScaler()

trained = sc.fit(y1_X_train)

y1_X_train = trained.transform(y1_X_train)
y1_X_test = trained.transform(y1_X_test)
y1_data_orig_train.features = y1_X_train
y1_data_orig_test.features = y1_X_test


# In[7]:


pca = PCA(n_components=2)

trained = pca.fit(y1_X_train)
y1_X_train = trained.transform(y1_X_train)
y1_X_test = trained.transform(y1_X_test)
y1_data_orig_test.features = y1_X_test


# In[8]:


y1_model = RandomForestClassifier()
y1_mdl = y1_model.fit(y1_X_train, y1_y_train)


# In[9]:


plot_model_performance(y1_mdl, y1_X_test, y1_y_test)

