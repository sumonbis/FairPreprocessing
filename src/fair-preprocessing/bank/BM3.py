#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/bank3-'
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

# y2_df = df.copy()
# Create a one-hot encoding of the categorical variables.
cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

for feature in cat_feat:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])

y1_df = df.copy()

def duration(dataframe):
    q1 = dataframe['duration'].quantile(0.25)
    q2 = dataframe['duration'].quantile(0.50)
    q3 = dataframe['duration'].quantile(0.75)
    dataframe.loc[(dataframe['duration'] <= q1), 'duration'] = 1
    dataframe.loc[(dataframe['duration'] > q1) & (dataframe['duration'] <= q2), 'duration'] = 2
    dataframe.loc[(dataframe['duration'] > q2) & (dataframe['duration'] <= q3), 'duration'] = 3
    dataframe.loc[(dataframe['duration'] > q3), 'duration'] = 4 
    print (q1, q2, q3)
    return dataframe
df = duration(df)

df.loc[(df['pdays'] == 999), 'pdays'] = 1
df.loc[(df['pdays'] > 0) & (df['pdays'] <= 10), 'pdays'] = 2
df.loc[(df['pdays'] > 10) & (df['pdays'] <= 20), 'pdays'] = 3
df.loc[(df['pdays'] > 20) & (df['pdays'] != 999), 'pdays'] = 4 

df.loc[(df['euribor3m'] < 1), 'euribor3m'] = 1
df.loc[(df['euribor3m'] > 1) & (df['euribor3m'] <= 2), 'euribor3m'] = 2
df.loc[(df['euribor3m'] > 2) & (df['euribor3m'] <= 3), 'euribor3m'] = 3
df.loc[(df['euribor3m'] > 3) & (df['euribor3m'] <= 4), 'euribor3m'] = 4
df.loc[(df['euribor3m'] > 4), 'euribor3m'] = 5

df['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)
y1_df['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)


# In[3]:



pro_att_name = ['age'] # ['race', 'sex']
priv_class = [1] # ['White', 'Male']
reamining_cat_feat = []
seed = randrange(100)

y2_data_orig, y2_X, y2_y = load_bank_data(df, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_train, y2_data_orig_test = y2_data_orig.split([0.7], shuffle=True, seed=seed)

y2_X_train = y2_data_orig_train.features
y2_y_train = y2_data_orig_train.labels.ravel()
y2_X_test = y2_data_orig_test.features
y2_y_test = y2_data_orig_test.labels.ravel()

y1_data_orig, y1_X, y1_y = load_bank_data(y1_df, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_train, y1_data_orig_test = y1_data_orig.split([0.7], shuffle=True, seed=seed)

y1_X_train = y1_data_orig_train.features
y1_y_train = y1_data_orig_train.labels.ravel()
y1_X_test = y1_data_orig_test.features
y1_y_test = y1_data_orig_test.labels.ravel()


sc2 = StandardScaler()
y2_X_train = sc2.fit_transform(y2_X_train)
y2_X_test = sc2.fit_transform(y2_X_test)
y2_data_orig_train.features = y2_X_train
y2_data_orig_test.features = y2_X_test

y1_X_train = sc2.fit_transform(y1_X_train)
y1_X_test = sc2.fit_transform(y1_X_test)
y1_data_orig_train.features = y1_X_train
y1_data_orig_test.features = y1_X_test

from sklearn.ensemble import GradientBoostingClassifier

y2_gbc = GradientBoostingClassifier()
y2_mdl = y2_gbc.fit(y2_X_train, y2_y_train)

y1_gbc = GradientBoostingClassifier()
y1_mdl = y1_gbc.fit(y1_X_train, y1_y_train)



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

stage = 'Custom(feature)'
model_name = 'bank3'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)
    

