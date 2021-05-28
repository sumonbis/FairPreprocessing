#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *


# In[2]:


f_path = "../../data/compas/compas.csv"
df = pd.read_csv(f_path)

# Create a one-hot encoding of the categorical variables.
# df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')

## Basic data cleaning
df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-
                          pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)
df['length_of_stay'].fillna(df['length_of_stay'].mode()[0], inplace = True)

df.drop('c_jail_in', axis=1, inplace=True)
df.drop('c_jail_out', axis=1, inplace=True)
# Recode sex and race
def group_race(x):
    if x == "Caucasian":
        return 1.0
    else:
        return 0.0
df['sex'] = df['sex'].replace({'Female': 1.0, 'Male': 0.0})
df['c_charge_degree'] = df['c_charge_degree'].replace({'F': 1.0, 'M': 0.0})
df['race'] = df['race'].apply(lambda x: group_race(x))


# In[3]:


############### Pipeline ###############
df = df[['sex','age','c_charge_degree','race','score_text',
             'priors_count','days_b_screening_arrest','decile_score',
             'is_recid','two_year_recid','length_of_stay']]


# In[4]:


df = df.loc[(df['days_b_screening_arrest'] <= 30)]
df = df.loc[(df['days_b_screening_arrest'] >= -30)]

df = df.loc[(df['is_recid'] != -1)]
df = df.loc[(df['c_charge_degree'] != "O")]
df = df.loc[(df['score_text'] != 'N/A')]


# In[5]:


##### Pipeline #######
# Stage 1
# y1_df = df.copy()
impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='mean')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])
featurizer1 = ColumnTransformer(transformers=[
        ('impute1_and_onehot', impute1_and_onehot, ['is_recid'])
    ], remainder='passthrough')

trans1 = pd.DataFrame(featurizer1.fit_transform(df[['is_recid']]), index=df.index, columns=['is_recid_no', 'is_recid_yes'])
df = pd.concat([df, trans1], axis=1)
df.drop('is_recid', axis=1, inplace=True)


# In[6]:


# Stage 2
impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')), # median, most_frequent,constant
                            ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))]) # n_bins has effect
featurizer2 = ColumnTransformer(transformers=[
        ('impute2_and_bin', impute2_and_bin, ['age'])
    ], remainder='passthrough')

df[['age']] = featurizer2.fit_transform(df[['age']])


# In[7]:

# Stage 3
## Binarizer
df = df.replace('Medium', "Low")
df['score_text'] = LabelEncoder().fit_transform(df['score_text'])


# In[8]:


seed = randrange(100)
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) #

pro_att_name = ['race']
priv_class = [1]
reamining_cat_feat = []

y1_data_orig_train, y1_X_train, y1_y_train = load_compas_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_compas_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)
y1_test_df = y1_data_orig_test.copy()

y1_model = LogisticRegression()
y1_mdl = y1_model.fit(y1_X_train, y1_y_train)


plot_model_performance(y1_mdl, y1_X_test, y1_y_test)
