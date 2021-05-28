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

# Create a one-hot encoding of the categorical variables.
cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

## Implement label encoder instead of one-hot encoder
for feature in cat_feat:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])


# In[3]:


seed = randrange(100)
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) # 

pro_att_name = ['race'] # ['race', 'sex']
priv_class = ['White'] # ['White', 'Male']
reamining_cat_feat = []

y1_data_orig_train, y1_X_train, y1_y_train = load_adult_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_adult_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)


# In[4]:


sc = StandardScaler()
trained = sc.fit(y1_X_train)
y1_X_train = trained.transform(y1_X_train)
y1_X_test = trained.transform(y1_X_test)

y1_data_orig_train.features = y1_X_train
y1_data_orig_test.features = y1_X_test


# In[5]:


pca = PCA(n_components=14)

trained = pca.fit(y1_X_train)
y1_X_train = trained.transform(y1_X_train)
y1_X_test = trained.transform(y1_X_test)
y1_data_orig_test.features = y1_X_test


# In[6]:


from catboost import CatBoostClassifier
y1_model = CatBoostClassifier()
y1_mdl = y1_model.fit(y1_X_train, y1_y_train)


# In[7]:


plot_model_performance(y1_mdl, y1_X_test, y1_y_test)

