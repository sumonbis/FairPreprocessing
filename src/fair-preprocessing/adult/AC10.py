#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/adult10-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


train_path = '../../../data/adult/adult.data'
test_path = '../../../data/adult/adult.test'

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


# In[3]:


##### Drop na values
# dropped = df.dropna()
# count = df.shape[0] - dropped.shape[0]
# print("Missing Data: {} rows removed.".format(count))
# df = dropped

y1_df = df.copy()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df['workclass'] = imputer.fit_transform(df[['workclass']]).ravel()
df['occupation'] = imputer.fit_transform(df[['occupation']]).ravel()
df['native-country'] = imputer.fit_transform(df[['native-country']]).ravel()


y1_df["workclass"] = y1_df["workclass"].fillna("X")
y1_df["occupation"] = y1_df["occupation"].fillna("X")
y1_df["native-country"] = y1_df["native-country"].fillna("x")


# nested_categorical_feature_transformation = Pipeline(steps=[
#         ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
# #         ('encode', OneHotEncoder(handle_unknown='ignore'))
#     ])


# In[4]:


# Create a one-hot encoding of the categorical variables.
cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')

# for feature in cat_feat:
#     le = LabelEncoder()
#     y2_df[feature] = le.fit_transform(y2_df[feature])
    
# for feature in cat_feat:
#     le = LabelEncoder()
#     y1_df[feature] = le.fit_transform(y1_df[feature])


# In[5]:


seed = randrange(100)
y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = seed, stratify=df['income-per-year']) 
y1_train, y1_test = train_test_split(y1_df, test_size = 0.3, random_state = seed) # stratify = df['income-per-year']

pro_att_name = ['race'] # ['race', 'sex']
priv_class = ['White'] # ['White', 'Male']
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_adult_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_adult_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)

y1_data_orig_train, y1_X_train, y1_y_train = load_adult_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_adult_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)


# In[6]:



y2_model = DecisionTreeClassifier()
y2_mdl = y2_model.fit(y2_X_train, y2_y_train)

y1_model = DecisionTreeClassifier()
y1_mdl = y1_model.fit(y1_X_train, y1_y_train)


# In[7]:


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

stage = 'Imputation'
model_name = 'adult10'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)

