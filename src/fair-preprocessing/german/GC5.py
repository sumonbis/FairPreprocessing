#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/german5-'
# Path(dir).mkdir(parents=True, exist_ok=True)

d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


filepath = '../../../data/german/german.data'
column_names = ['status', 'month', 'credit_history',
            'purpose', 'credit_amount', 'savings', 'employment',
            'investment_as_income_percentage', 'personal_status',
            'other_debtors', 'residence_since', 'property', 'age',
            'installment_plans', 'housing', 'number_of_credits',
            'skill_level', 'people_liable_for', 'telephone',
            'foreign_worker', 'credit']
na_values=[]
df = pd.read_csv(filepath, sep=' ', header=None, names=column_names,na_values=na_values)
df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
df = german_custom_preprocessing(df)
feat_to_drop = ['personal_status']
df = df.drop(feat_to_drop, axis=1)

cat_feat = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
# num_feat = ['residence_since', 'age', 'investment_as_income_percentage', 'credit_amount', 'number_of_credits', 'people_liable_for', 'month']


# In[3]:


seed = randrange(100)
y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['race']
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) # 

pro_att_name = ['age'] # ['sex', 'age']
priv_class = [1]
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_german_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_german_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)

y1_data_orig_train, y1_X_train, y1_y_train = load_german_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_german_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)

y1_test_df = y2_data_orig_test.copy()


# In[4]:


sc = StandardScaler()

trained = sc.fit(y2_X_train)
y2_X_train = trained.transform(y2_X_train)
y2_X_test = trained.transform(y2_X_test)

y2_data_orig_train.features = y2_X_train
y2_data_orig_test.features = y2_X_test

# trained = sc.fit(y1_X_train)
# y1_X_train = trained.transform(y1_X_train)
# y1_X_test = trained.transform(y1_X_test)

# y1_data_orig_train.features = y1_X_train
# y1_data_orig_test.features = y1_X_test


# In[5]:



from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
param_grid_forest = [
    {
        'n_estimators': [5, 10, 20, 50],        
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5],
        'max_depth': [None, 5, 10, 15],
        'min_samples_leaf': [1, 5],
        'bootstrap': [True, False]
    }
]

#GridSearch
forest_clf = RandomForestClassifier()
grid_search = GridSearchCV(forest_clf, param_grid_forest, cv=5, scoring='roc_auc', verbose=0)
grid_search.fit(y2_X_train, y2_y_train)
best_model = grid_search.best_estimator_

log_reg = LogisticRegression()
svm_clf = SVC(probability=True)
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier()
tree_clf = DecisionTreeClassifier()

voting = VotingClassifier(
    estimators=[('lr', log_reg), ('best_rf', best_model), ('svc', svm_clf), 
                ('knn', knn_clf), ('mlp', mlp_clf), ('tree', tree_clf)],
    voting='soft'
)

y2_model = voting
y2_mdl = y2_model.fit(y2_X_train, y2_y_train)


forest_clf = RandomForestClassifier()
grid_search = GridSearchCV(forest_clf, param_grid_forest, cv=5, scoring='roc_auc', verbose=0)
grid_search.fit(y1_X_train, y1_y_train)
best_model = grid_search.best_estimator_

log_reg = LogisticRegression()
svm_clf = SVC(probability=True)
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier()
tree_clf = DecisionTreeClassifier()

voting = VotingClassifier(
    estimators=[('lr', log_reg), ('best_rf', best_model), ('svc', svm_clf), 
                ('knn', knn_clf), ('mlp', mlp_clf), ('tree', tree_clf)],
    voting='soft'
)

y1_model = voting
y1_mdl = y1_model.fit(y1_X_train, y1_y_train)


# In[6]:


# plot_model_performance(y2_mdl, y2_X_test, y2_y_test)
y1_pred, y1_fair = get_fair_metrics_and_plot('filename', y1_data_orig_test, y1_mdl)
y2_pred, y2_fair = get_fair_metrics_and_plot('filename', y2_data_orig_test, y2_mdl)


y1_fair = y1_fair.drop(['DI', 'CNT', 'TI'], axis=1)
y2_fair = y2_fair.drop(['DI', 'CNT', 'TI'], axis=1)
CVR, CVD, AVR_EOD, AVD_EOD, AVR_SPD, AVD_SPD, AVD_AOD, AV_ERD = compute_new_metrics(y1_test_df, y1_pred, y2_pred)
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
model_name = 'german5'
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




