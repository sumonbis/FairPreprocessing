#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/adult6-'
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

##### Drop na values
dropped = df.dropna()
count = df.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df = dropped


# In[3]:


y1_df = df.copy()

df["marital-status"] = df["marital-status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
df["marital-status"] = df["marital-status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
df["marital-status"] = df["marital-status"].map({"Married":0, "Single":1})
df["marital-status"] = df["marital-status"]

excluded_feat = ["workclass","education","occupation","relationship","native-country"]

df.drop(labels=excluded_feat, axis=1, inplace=True)

y1_df.drop(labels=excluded_feat, axis=1, inplace=True)


# In[4]:


# Create a one-hot encoding of the categorical variables.
cat_feat = ['age', 'hours-per-week', 'sex']
ccat_feat = ['age', 'hours-per-week', 'sex', 'marital-status']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
y1_df = pd.get_dummies(y1_df, columns=ccat_feat, prefix_sep='=')

# y2_cat_feat = ["workclass","education","occupation","relationship","native-country", 'age', 'hours-per-week', 'sex']
# cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


# In[5]:


seed = randrange(100)
y2_train, y2_test = train_test_split(y1_df, test_size = 0.3, random_state = seed) # stratify=df['race']
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) # 

pro_att_name = ['race'] # ['race', 'sex']
priv_class = ['White'] # ['White', 'Male']
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_adult_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_adult_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)

y1_data_orig_train, y1_X_train, y1_y_train = load_adult_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_adult_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)


# In[6]:


from xgboost import XGBClassifier
# n_estimators=1600,learning_rate=0.05
y2_model = XGBClassifier() 
y2_mdl = y2_model.fit(y2_X_train, y2_y_train, verbose=False)

y1_model = XGBClassifier()
y1_mdl = y1_model.fit(y1_X_train, y1_y_train, verbose=False)


# In[7]:


# plot_model_performance(y2_mdl, y2_X_test, y2_y_test)
y1_pred, y1_fair = get_fair_metrics_and_plot('filename', y1_data_orig_test, y1_mdl)
y2_pred, y2_fair = get_fair_metrics_and_plot('filename', y2_data_orig_test, y2_mdl)


y1_fair = y1_fair.drop(['DI', 'CNT', 'TI'], axis=1)
y2_fair = y2_fair.drop(['DI', 'CNT', 'TI'], axis=1)
CVR, CVD, AVR_EOD, AVD_EOD, AVR_SPD, AVD_SPD, AVD_AOD, AV_ERD = compute_new_metrics(y1_data_orig_test, y1_pred, y2_pred)
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

stage = 'Custom(clean)'
model_name = 'adult6'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)
   

