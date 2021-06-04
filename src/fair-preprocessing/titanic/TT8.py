#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/titanic8-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


import re 
from sklearn.ensemble import RandomForestRegressor
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    #If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def fill_missing_age(df):
    
    #Feature set
    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp',
                 'TicketNumber', 'Title','Pclass','FamilySize',
                 'FsizeD','NameLength',"NlengthD",'Deck']]
    # Split sets into train and test
    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values
    test = age_df.loc[ (df.Age.isnull()) ]# null Ages
    
    # All age values are stored in a target array
    y = train.values[:, 0]
    
    # All the other values are stored in the feature array
    X = train.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])
    
    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df


# In[3]:


# Load data
train = pd.read_csv('../../../data/titanic/train.csv')
test = pd.read_csv('../../../data/titanic/test.csv')
df = train


# In[4]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

y1_df = df.copy()
## Custom(feature)
df["Embarked"] = df["Embarked"].fillna('C')
def fill_missing_fare(df):
    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()
    df["Fare"] = df["Fare"].fillna(median_fare)
    return df
df=fill_missing_fare(df)

df["Deck"]=df.Cabin.str[0]
df.Deck.fillna('Z', inplace=True)

df["FamilySize"] = df["SibSp"] + df["Parch"]+1

df.loc[df["FamilySize"] == 1, "FsizeD"] = 'singleton'
df.loc[(df["FamilySize"] > 1)  &  (df["FamilySize"] < 5) , "FsizeD"] = 'small'
df.loc[df["FamilySize"] >4, "FsizeD"] = 'large'

df["NameLength"] = df["Name"].apply(lambda x: len(x))
bins = [0, 20, 40, 57, 85]
group_names = ['short', 'okay', 'good', 'long']
df['NlengthD'] = pd.cut(df['NameLength'], bins, labels=group_names)

df["TicketNumber"] = df["Ticket"].str.extract('(\d{2,})', expand=True)
df["TicketNumber"] = df["TicketNumber"].apply(pd.to_numeric)
df.TicketNumber.fillna(df["TicketNumber"].median(), inplace=True)

titles = df["Name"].apply(get_title)
df["Title"] = titles
# Titles with very low cell counts to be combined to "rare" level
rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
df.loc[df["Title"] == "Mlle", "Title"] = 'Miss'
df.loc[df["Title"] == "Ms", "Title"] = 'Miss'
df.loc[df["Title"] == "Mme", "Title"] = 'Mrs'
df.loc[df["Title"] == "Dona", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Lady", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Countess", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Capt", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Col", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Don", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Major", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Rev", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Sir", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Jonkheer", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Dr", "Title"] = 'Rare Title'

labelEnc=LabelEncoder()
cat_vars=['Embarked','Sex',"Title","FsizeD","NlengthD",'Deck']
for col in cat_vars:
    df[col]=labelEnc.fit_transform(df[col])

df=fill_missing_age(df)

drop_column = ['PassengerId', 'Cabin', 'Ticket', 'Name', 'Parch']
df.drop(drop_column, axis=1, inplace = True)
    

std_scale = StandardScaler().fit(df[['Age', 'Fare']])
df[['Age', 'Fare']] = std_scale.transform(df[['Age', 'Fare']])


y1_df['Age'].fillna(y1_df['Age'].median(), inplace = True)
y1_df = y1_df.fillna({"Embarked": "S"})
y1_df['Cabin'].fillna(y1_df['Cabin'].mode(), inplace = True)

# One-hot encoder
cat_feat = ['Cabin', 'Ticket', 'Embarked']
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')

y1_df = y1_df.drop(['PassengerId'], axis = 1)
y1_df = y1_df.drop(['Name'], axis = 1)


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




y2_model = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=9,min_samples_split=6, min_samples_leaf=4)
y2_mdl = y2_model.fit(y2_X_train, y2_y_train)

y1_model = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=9,min_samples_split=6, min_samples_leaf=4)
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
model_name = 'titanic8'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)

