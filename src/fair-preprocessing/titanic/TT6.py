#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/titanic6-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


# Load data
train = pd.read_csv('../../../data/titanic/train.csv')
test = pd.read_csv('../../../data/titanic/test.csv')
df = train


# In[3]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

y1_df = df.copy()

df['Child'] = df['Age']<=10
df['Cabin_known'] = df['Cabin'].isnull() == False
df['Age_known'] = df['Age'].isnull() == False
df['Family'] = df['SibSp'] + df['Parch']
df['Alone']  = (df['SibSp'] + df['Parch']) == 0
df['Large_Family'] = (df['SibSp']>2) | (df['Parch']>3)
df['Deck'] = df['Cabin'].str[0]
df['Deck'] = df['Deck'].fillna(value='U')
df['Ttype'] = df['Ticket'].str[0]
df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
df['Fare_cat'] = pd.DataFrame(np.floor(np.log10(df['Fare'] + 1))).astype('int')
df['Bad_ticket'] = df['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])
df['Young'] = (df['Age']<=30) | (df['Title'].isin(['Master','Miss','Mlle']))
df['Shared_ticket'] = np.where(df.groupby('Ticket')['Name'].transform('count') > 1, 1, 0)
df['Ticket_group'] = df.groupby('Ticket')['Name'].transform('count')
df['Fare_eff'] = df['Fare']/df['Ticket_group']
df['Fare_eff_cat'] = np.where(df['Fare_eff']>16.0, 2, 1)
df['Fare_eff_cat'] = np.where(df['Fare_eff']<8.5,0,df['Fare_eff_cat'])
df['Age'].fillna(df['Age'].median(), inplace = True)
df = df.fillna({"Embarked": "S"})

cat_feat = ['Embarked', 'Deck', 'Title', 'Ttype',]
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


drop_column = ['Name', 'Ticket', 'Fare', 'Cabin']
df.drop(drop_column, axis=1, inplace = True)



y1_df['Age'].fillna(y1_df['Age'].median(), inplace = True)
y1_df = y1_df.fillna({"Embarked": "S"})
y1_df['Cabin'].fillna(y1_df['Cabin'].mode(), inplace = True)

# One-hot encoder
cat_feat = ['Cabin', 'Ticket', 'Embarked']
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')

y1_df = y1_df.drop(['PassengerId'], axis = 1)
y1_df = y1_df.drop(['Name'], axis = 1)


# In[4]:



seed = randrange(100)
y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['loan']
y1_train, y1_test = train_test_split(y1_df, test_size = 0.3, random_state = seed) # 

pro_att_name = ['Sex']
priv_class = [1]
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_titanic_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_titanic_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)

y1_data_orig_train, y1_X_train, y1_y_train = load_titanic_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_titanic_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)

import xgboost as xgb
y2_xgb = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scale_pos_weight=1)

y2_mdl = y2_xgb.fit(y2_X_train, y2_y_train)

y1_xgb = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scale_pos_weight=1)

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

stage = 'Custom(feature)'
model_name = 'titanic6'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)

