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
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
# num_feat = ['residence_since', 'age', 'investment_as_income_percentage', 'credit_amount', 'number_of_credits', 'people_liable_for', 'month']


# In[3]:


##### Pipeline
y1_df = df.copy()
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')
for f in cat_feat:
    label = LabelEncoder()
    df[f] = label.fit_transform(df[f])


# In[4]:


features = ['age', 'sex', 'employment', 'housing', 'savings',
       'number_of_credits', 'credit_amount', 'month', 'purpose', 'credit']
df = df[features]


# In[5]:


seed = randrange(100)
y1_train, y1_test = train_test_split(y1_df, test_size = 0.3, random_state = seed) # 

pro_att_name = ['age'] # ['sex', 'age']
priv_class = [1]
reamining_cat_feat = []

y1_data_orig_train, y1_X_train, y1_y_train = load_german_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_german_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)


# In[6]:


sc = StandardScaler()

trained = sc.fit(y1_X_train)
y1_X_train = trained.transform(y1_X_train)
y1_X_test = trained.transform(y1_X_test)

y1_data_orig_train.features = y1_X_train
y1_data_orig_test.features = y1_X_test


# In[7]:


pca = PCA(n_components=None)

trained_f = pca.fit(y1_X_train)
y1_X_train = trained_f.transform(y1_X_train)
y1_X_test = trained_f.transform(y1_X_test)
y1_data_orig_test.features = y1_X_test


# In[8]:


rfc_clf = RandomForestClassifier()
# params = {'n_estimators':[25,50,100,150,200,500],'max_depth':[0.5,1,5,10],'random_state':[1,10,20,42],
#           'n_jobs':[1,2]}
params = {'n_estimators':[100,150,200],'max_depth':[5,10],'random_state':[1, 42], 'n_jobs':[1]}
y1_grid_search_cv = GridSearchCV(rfc_clf, params, scoring='precision')
y1_grid_search_cv.fit(y1_X_train, y1_y_train)
y1_rfc_clf = y1_grid_search_cv.best_estimator_
y1_mdl = y1_rfc_clf.fit(y1_X_train, y1_y_train)


# In[9]:


plot_model_performance(y1_mdl, y1_X_test, y1_y_test)

