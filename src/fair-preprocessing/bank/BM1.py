#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/bank1-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


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

# Create a one-hot encoding of the categorical variables.
cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
y2_df = df.copy()
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

labelencoder_X = LabelEncoder()
y2_df['job']      = labelencoder_X.fit_transform(y2_df['job']) 
y2_df['marital']  = labelencoder_X.fit_transform(y2_df['marital']) 
y2_df['education']= labelencoder_X.fit_transform(y2_df['education']) 
y2_df['default']  = labelencoder_X.fit_transform(y2_df['default']) 
y2_df['housing']  = labelencoder_X.fit_transform(y2_df['housing']) 
y2_df['loan']     = labelencoder_X.fit_transform(y2_df['loan'])
y2_df['contact']     = labelencoder_X.fit_transform(y2_df['contact']) 
y2_df['month']       = labelencoder_X.fit_transform(y2_df['month']) 
y2_df['day_of_week'] = labelencoder_X.fit_transform(y2_df['day_of_week'])
y2_df['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)

def duration(data):
    data.loc[data['duration'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']    = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration'] = 4
    data.loc[data['duration']  > 644.5, 'duration'] = 5

    return data

# y2_df = df.copy()
y2_df = duration(y2_df)


# In[3]:


# y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = 0) #, stratify=df['race']
# y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = 0)


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

y1_data_orig, y1_X, y1_y = load_bank_data(y2_df, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_train, y1_data_orig_test = y1_data_orig.split([0.7], shuffle=True, seed=seed)

y1_X_train = y1_data_orig_train.features
y1_y_train = y1_data_orig_train.labels.ravel()
y1_X_test = y1_data_orig_test.features
y1_y_test = y1_data_orig_test.labels.ravel()



from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sc = StandardScaler()
y2_X_train = sc.fit_transform(y2_X_train)
y2_X_test = sc.fit_transform(y2_X_test)
y2_data_orig_train.features = y2_X_train
y2_data_orig_test.features = y2_X_test

y2_xgb = XGBClassifier()
y2_mdl = y2_xgb.fit(y2_X_train, y2_y_train)

#     y1_X_train = sc.fit_transform(y1_X_train)
#     y1_X_test = sc.transform(y1_X_test)
#     y1_data_orig_train.features = y1_X_train
#     y1_data_orig_test.features = y1_X_test

y1_xgb = XGBClassifier()
y1_mdl = y1_xgb.fit(y1_X_train, y1_y_train)



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

stage = 'StandardScaler'
model_name = 'bank1'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)

