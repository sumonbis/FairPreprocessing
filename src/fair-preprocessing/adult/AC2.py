#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/adult2-'
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

seed = randrange(100)
y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['race']
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) # 


# In[3]:


##### Process na values
dropped = y1_train.dropna()
count = y1_train.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
y1_train = dropped

dropped = y1_test.dropna()
count = y1_test.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
y1_test = dropped

# Fill Missing Category Entries
y2_train["workclass"] = y2_train["workclass"].fillna("X")
y2_train["occupation"] = y2_train["occupation"].fillna("X")
y2_train["native-country"] = y2_train["native-country"].fillna("United-States")

# y2_test["workclass"] = y2_test["workclass"].fillna("X")
# y2_test["occupation"] = y2_test["occupation"].fillna("X")
# y2_test["native-country"] = y2_test["native-country"].fillna("United-States")
dropped = y2_test.dropna()
count = y2_test.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
y2_test = dropped


# Create a one-hot encoding of the categorical variables.
cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
# y1_df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

# y2_train = pd.get_dummies(y2_train, columns=cat_feat, prefix_sep='=')
# y1_train = pd.get_dummies(y1_train, columns=cat_feat, prefix_sep='=')

# y2_test = pd.get_dummies(y2_test, columns=cat_feat, prefix_sep='=')
# y1_test = pd.get_dummies(y1_test, columns=cat_feat, prefix_sep='=')

## Implement label encoder instead of one-hot encoder
for feature in cat_feat:
    le = LabelEncoder()
    y2_train[feature] = le.fit_transform(y2_train[feature])
    y2_test[feature] = le.transform(y2_test[feature])

for feature in cat_feat:
    le = LabelEncoder()
    y1_train[feature] = le.fit_transform(y1_train[feature])
    y1_test[feature] = le.transform(y1_test[feature])


# In[4]:


pro_att_name = ['race'] # ['race', 'sex']
priv_class = ['White'] # ['White', 'Male']
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_adult_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_adult_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)

y1_data_orig_train, y1_X_train, y1_y_train = load_adult_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_adult_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)


# In[5]:


y2_model =  RandomForestClassifier(n_estimators=250,max_features=5)
y2_mdl = y2_model.fit(y2_X_train, y2_y_train)

y1_model =  RandomForestClassifier(n_estimators=250,max_features=5)
y1_mdl = y1_model.fit(y1_X_train, y1_y_train)


# In[6]:


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

stage = 'MissingValue'
model_name = 'adult2'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)

