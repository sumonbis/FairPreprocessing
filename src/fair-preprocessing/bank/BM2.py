#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/bank2-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


def custom_get_fair_metrics_and_plot(fname, data, model, model_aif=False):
    pred = model.predict(data).labels if model_aif else model.predict(data.features)
    pred = (pred >= 0.5) * 1
    fair = fair_metrics(fname, data, pred)
    return (pred, fair)


# In[3]:


file_path = '../../../data/bank/bank-additional-full.csv'

column_names = []
na_values=['unknown']

df = pd.read_csv(file_path, sep=';', na_values=na_values)

#### Drop na values
dropped = df.dropna()
count = df.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df = dropped

df['age'] = df['age'].apply(lambda x: np.float(x >= 25))

## Feature selection
# features_to_keep = []
# df = df[features_to_keep]
y2_df = df.copy()
# Create a one-hot encoding of the categorical variables.
cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

for feature in cat_feat:
    le = LabelEncoder()
    y2_df[feature] = le.fit_transform(y2_df[feature])


# In[4]:



pro_att_name = ['age'] # ['race', 'sex']
priv_class = [1] # ['White', 'Male']
reamining_cat_feat = []
seed = randrange(100)

y2_data_orig, y2_X, y2_y = load_bank_data(y2_df, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_train, y2_data_orig_test = y2_data_orig.split([0.7], shuffle=True, seed=seed)

y2_X_train = y2_data_orig_train.features
y2_y_train = y2_data_orig_train.labels.ravel()
y2_X_test = y2_data_orig_test.features
y2_y_test = y2_data_orig_test.labels.ravel()

y1_data_orig, y1_X, y1_y = load_bank_data(df, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_train, y1_data_orig_test = y1_data_orig.split([0.7], shuffle=True, seed=seed)

y1_X_train = y1_data_orig_train.features
y1_y_train = y1_data_orig_train.labels.ravel()
y1_X_test = y1_data_orig_test.features
y1_y_test = y1_data_orig_test.labels.ravel()

import lightgbm as lgb
from xgboost.sklearn import XGBClassifier

y2_lgb_train = lgb.Dataset(data=y2_X_train, label=y2_y_train,  free_raw_data=False)
y2_lgb_eval = lgb.Dataset(data=y2_X_test, label=y2_y_test, reference=y2_lgb_train,  free_raw_data=False)
y2_evals_result={}
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

y2_md = XGBClassifier()
y2_mdl = lgb.train(params,
                y2_lgb_train,
                valid_sets = y2_lgb_eval,
                num_boost_round= 150,
                early_stopping_rounds= 25,
                evals_result=y2_evals_result)


y1_lgb_train = lgb.Dataset(data=y1_X_train, label=y1_y_train,  free_raw_data=False)
y1_lgb_eval = lgb.Dataset(data=y1_X_test, label=y1_y_test, reference=y1_lgb_train,  free_raw_data=False)
y1_evals_result={}

y1_md = XGBClassifier()
y1_mdl = lgb.train(params,
                y1_lgb_train,
                valid_sets = y1_lgb_eval,
                num_boost_round= 150,
                early_stopping_rounds= 25,
                evals_result=y1_evals_result)



# plot_model_performance(y2_mdl, y2_X_test, y2_y_test)
y1_pred, y1_fair = custom_get_fair_metrics_and_plot('filename', y1_data_orig_test, y1_mdl)
y2_pred, y2_fair = custom_get_fair_metrics_and_plot('filename', y2_data_orig_test, y2_mdl)



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

stage = 'LabelEncoder'
model_name = 'bank2'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)
    
  

