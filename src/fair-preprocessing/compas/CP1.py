#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/compas-'

d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


f_path = "../../../data/compas/compas.csv"
df = pd.read_csv(f_path)

# Create a one-hot encoding of the categorical variables.
# df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')

## Basic data cleaning
df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-
                          pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)
df['length_of_stay'].fillna(df['length_of_stay'].mode()[0], inplace = True)


df.drop('c_jail_in', axis=1, inplace=True)
df.drop('c_jail_out', axis=1, inplace=True)
# del data['c_jail_in']
# del data['c_jail_out']
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

# y1_data = data[['sex','age','c_charge_degree','race','score_text',
#              'priors_count','days_b_screening_arrest','decile_score',
#              'is_recid','two_year_recid','length_of_stay',
#              'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'c_charge_desc']]


# In[4]:


# data = data.dropna(subset=['days_b_screening_arrest'])
# y1_df = df.copy()

df = df.loc[(df['days_b_screening_arrest'] <= 30)]
df = df.loc[(df['days_b_screening_arrest'] >= -30)]

df = df.loc[(df['is_recid'] != -1)]
df = df.loc[(df['c_charge_degree'] != "O")]
df = df.loc[(df['score_text'] != 'N/A')]

# y1_df['days_b_screening_arrest'].fillna(y1_df['days_b_screening_arrest'].mean(), inplace = True)
#     data['c_jail_out'] = pd.to_datetime(data['c_jail_out'])
#     data['c_jail_in'] = pd.to_datetime(data['c_jail_in'])


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


# impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='mean')) ])
# featurizer1 = ColumnTransformer(transformers=[
#         ('impute1_and_onehot', impute1_and_onehot, ['is_recid'])
#     ], remainder='passthrough')
# trans1 = pd.DataFrame(featurizer1.fit_transform(y1_df[['is_recid']]), index=y1_df.index, columns=['is_recid'])

# y1_df.drop('is_recid', axis=1, inplace=True)
# y1_df = pd.concat([y1_df, trans1], axis=1)
# y1_df['is_recid'] = LabelEncoder().fit_transform(y1_df['is_recid'])


# In[6]:


# Stage 2
# y1_df = df.copy()
impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')), # median, most_frequent,constant
                            ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))]) # n_bins has effect
featurizer2 = ColumnTransformer(transformers=[
        ('impute2_and_bin', impute2_and_bin, ['age'])
    ], remainder='passthrough')

df[['age']] = featurizer2.fit_transform(df[['age']])


# impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')), # median, most_frequent,constant
#                             ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))]) # n_bins has effect
# featurizer2 = ColumnTransformer(transformers=[
#         ('impute2_and_bin', impute2_and_bin, ['age'])
#     ], remainder='passthrough')


# y1_df[['age']] = featurizer2.fit_transform(y1_df[['age']])



# y1_data['c_charge_desc'] = y1_data['c_charge_desc'].astype(str)
# y1_data['c_charge_desc'] = LabelEncoder().fit_transform(y1_data['c_charge_desc'])


# In[7]:


# Stage 3

y1_df = df.copy()
## Binarizer
df = df.replace('Medium', "Low")
df['score_text'] = LabelEncoder().fit_transform(df['score_text'])

cat_feat = ['score_text']
# y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')
# y1_df['score_text'] = LabelEncoder().fit_transform(y1_df['score_text'])

# y1_df = y1_df.replace('Medium', "Low")
y1_df['score_text'] = LabelEncoder().fit_transform(y1_df['score_text'])


# In[8]:


seed = randrange(100)
y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['race']
y1_train, y1_test = train_test_split(y1_df, test_size = 0.3, random_state = seed) # 

pro_att_name = ['race']
priv_class = [1]
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_compas_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_compas_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)

y1_data_orig_train, y1_X_train, y1_y_train = load_compas_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_compas_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)
y1_test_df = y1_data_orig_test.copy()


y2_model = LogisticRegression()
y2_mdl = y2_model.fit(y2_X_train, y2_y_train)

y1_model = LogisticRegression()
y1_mdl = y1_model.fit(y1_X_train, y1_y_train)


# plot_model_performance(y2_mdl, y2_X_test, y2_y_test)
y1_pred, y1_fair = get_fair_metrics_and_plot('filename', y1_data_orig_test, y1_mdl)
y2_pred, y2_fair = get_fair_metrics_and_plot('filename', y2_data_orig_test, y2_mdl)



y1_fair = y1_fair.drop(['DI', 'CNT', 'TI'], axis=1)
y2_fair = y2_fair.drop(['DI', 'CNT', 'TI'], axis=1)
CVR, CVD, AVR_EOD, AVD_EOD, AVR_SPD, AVD_SPD, AVD_AOD, AV_ERD = compute_new_metrics(y2_data_orig_test, y1_pred, y2_pred)
row_y1 = y1_fair.iloc[[0]].values[0].tolist()
row_y2 = y2_fair.iloc[[0]].values[0].tolist()
diff = []
# diff.append(CVR)
# diff.append(CVD)
diff.append(AVD_SPD)
diff.append(AVD_EOD)
diff.append(AVD_AOD)
diff.append(AV_ERD)

for i in range(len(row_y2)):
    if(i < 2):
        change = row_y2[i] - row_y1[i]
    else:
        break;
    diff.append(change)

stage = 'Binarizer'
model_name = 'compas'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)
    

