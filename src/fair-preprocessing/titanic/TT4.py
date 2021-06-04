#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/titanic4-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'


# In[3]:


# Load data
train = pd.read_csv('../../../data/titanic/train.csv')
test = pd.read_csv('../../../data/titanic/test.csv')
df = train


# In[4]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

## Imputation
df[ 'Age' ] = df.Age.fillna( df.Age.mean() )
df[ 'Fare' ] = df.Fare.fillna( df.Fare.mean() )
## filna(-1)

    
## Custom(feature)
title = pd.DataFrame()
title[ 'Title' ] = df[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
title[ 'Title' ] = title.Title.map( Title_Dictionary )
df[ 'Title' ] = title[ 'Title' ]
df[ 'Ticket' ] = df[ 'Ticket' ].map( cleanTicket )
df[ 'Cabin' ] = df.Cabin.fillna( 'U' )
df[ 'FamilySize' ] = df[ 'Parch' ] + df[ 'SibSp' ] + 1
df[ 'Family_Single' ] = df[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
df[ 'Family_Small' ]  = df[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
df[ 'Family_Large' ]  = df[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )


# Basic
# One-hot encoder
cat_feat = ['Title', 'Ticket', 'Cabin'] #   'Ticket', 'Embarked'
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

drop_column = ['Embarked', 'PassengerId', 'Name']
df.drop(drop_column, axis=1, inplace = True)

# Basic
# One-hot encoder
# cat_feat = ['Ticket', 'Cabin'] #   'Ticket', 'Embarked'
# y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')

# drop_column = ['Embarked', 'PassengerId', 'Name']
# y1_df.drop(drop_column, axis=1, inplace = True)


# In[5]:



seed = randrange(100)
y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['loan']
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) # 


pro_att_name = ['Sex']
priv_class = [1]
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_titanic_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_titanic_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)

y1_data_orig_train, y1_X_train, y1_y_train = load_titanic_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_titanic_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)


## FeatureSelection
from sklearn.feature_selection import RFECV
model = LogisticRegression()
rfecv = RFECV( estimator = model , step = 1 , cv = 2 , scoring = 'accuracy' )
trained_rfecv = rfecv.fit( y2_X_train , y2_y_train )
y2_X_train = trained_rfecv.transform(y2_X_train)
y2_X_test = trained_rfecv.transform(y2_X_test)
y2_data_orig_train.features = y2_X_train
y2_data_orig_test.features = y2_X_test



y2_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

y2_mdl = y2_model.fit(y2_X_train, y2_y_train) 

y1_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

y1_mdl = y1_model.fit(y1_X_train, y1_y_train) 


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

stage = 'RFECV'
model_name = 'titanic4'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)


# In[ ]:




