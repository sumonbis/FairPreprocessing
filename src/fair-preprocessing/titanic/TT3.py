#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/titanic3-'
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


def name_converted(feature):
    result = ''
    if feature in ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col', 'Rev', 'Dona', 'Dr']:
        result = 'rare'
    elif feature in ['Ms', 'Mlle']:
        result = 'Miss'
    elif feature == 'Mme':
        result = 'Mrs'
    else:
        result = feature
    return result
def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a
def fare_group(fare):
    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a


# In[4]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

### Feature
y1_df = df.copy()

## Custom(feature)
df['title'] = [i.split('.')[0].split(',')[1].strip() for i in df.Name]
df.title = df.title.map(name_converted)

## Family_size seems like a good feature to create
df['family_size'] = df.SibSp + df.Parch+1
df['is_alone'] = [1 if i<2 else 0 for i in df.family_size]
df['family_group'] = df['family_size'].map(family_group)

df['calculated_fare'] = df.Fare/df.family_size
df['fare_group'] = df['calculated_fare'].map(fare_group)

df.drop(['Ticket'], axis=1, inplace=True)
df.drop(['PassengerId'], axis=1, inplace=True)
df.drop(['Name'], axis=1, inplace=True)


## Imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(missing_values=np.NaN)
df['Age'] = imputer.fit_transform(df[['Age']]).ravel()

df = pd.get_dummies(df, columns=['title',"Pclass", 'Cabin','Embarked', 'family_group', 'fare_group'], drop_first=False)
# y1_df = pd.get_dummies(y1_df, columns=['title',"Pclass", 'Cabin','Embarked', 'family_group', 'fare_group'], drop_first=False)

y1_df.drop(['Ticket'], axis=1, inplace=True)
y1_df.drop(['PassengerId'], axis=1, inplace=True)
y1_df.drop(['Name'], axis=1, inplace=True)

imputer = KNNImputer(missing_values=np.NaN)
y1_df['Age'] = imputer.fit_transform(y1_df[['Age']]).ravel()

y1_df = pd.get_dummies(y1_df, columns=["Pclass", 'Cabin','Embarked'], drop_first=False)
# y1_df = pd.get_dummies(y1_df, columns=['title',"Pclass", 'Cabin','Embarked', 'family_group', 'fare_group'], drop_first=False)


# In[5]:



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




from sklearn.ensemble import BaggingClassifier
y2_model = BaggingClassifier(base_estimator=None, bootstrap=True, bootstrap_features=False,
                  max_features=1.0, max_samples=1.0, n_estimators=180,
                  n_jobs=None, oob_score=False, random_state=None, verbose=0,
                  warm_start=False)
y2_mdl = y2_model.fit(y2_X_train, y2_y_train)

y1_model = BaggingClassifier(base_estimator=None, bootstrap=True, bootstrap_features=False,
                  max_features=1.0, max_samples=1.0, n_estimators=180,
                  n_jobs=None, oob_score=False, random_state=None, verbose=0,
                  warm_start=False)
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

stage = 'Custom(feature)'
model_name = 'titanic3'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)

