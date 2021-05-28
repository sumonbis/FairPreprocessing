from aif360.datasets import StandardDataset
import numpy as np

def custom_preprocessing(df):
    def group_race(x):
        if x == "White":
            return 1.0
        else:
            return 0.0
    # Recode sex and race
    df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
    df['race'] = df['race'].apply(lambda x: group_race(x))

    return df

def german_custom_preprocessing(df):
    def group_credit_hist(x):
        if x in ['A30', 'A31', 'A32']:
            return 'None/Paid'
        elif x == 'A33':
            return 'Delay'
        elif x == 'A34':
            return 'Other'
        else:
            return 'NA'

    def group_employ(x):
        if x == 'A71':
            return 'Unemployed'
        elif x in ['A72', 'A73']:
            return '1-4 years'
        elif x in ['A74', 'A75']:
            return '4+ years'
        else:
            return 'NA'

    def group_savings(x):
        if x in ['A61', 'A62']:
            return '<500'
        elif x in ['A63', 'A64']:
            return '500+'
        elif x == 'A65':
            return 'Unknown/None'
        else:
            return 'NA'

    def group_status(x):
        if x in ['A11', 'A12']:
            return '<200'
        elif x in ['A13']:
            return '200+'
        elif x == 'A14':
            return 'None'
        else:
            return 'NA'

    # status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
    #               'A92': 0.0, 'A95': 0.0}  # A91: male
    # df['sex'] = df['personal_status'].replace(status_map)
    status_map = {'A91': 1, 'A93': 1, 'A94': 1, 'A92': 0, 'A95': 0}  # 1: 'male'
    df['sex'] = df['personal_status'].replace(status_map)

    # group credit history, savings, and employment
    df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
    df['savings'] = df['savings'].apply(lambda x: group_savings(x))
    df['employment'] = df['employment'].apply(lambda x: group_employ(x))
    # df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
    df['status'] = df['status'].apply(lambda x: group_status(x))

    df.credit.replace([1, 2], [1, 0], inplace=True)

    return df


def load_german_data(df, protected_attribute_name, privileged_class, categorical_features):
    data_orig = StandardDataset(df,
                                label_name='credit',
                                favorable_classes=[1],
                                protected_attribute_names=protected_attribute_name,
                                privileged_classes=[privileged_class],
                                categorical_features=categorical_features,
                                #custom_preprocessing=german_custom_preprocessing
                                )
    X = data_orig.features
    y = data_orig.labels.ravel()
    return data_orig, X, y

def load_adult_data(df, protected_attribute_name, privileged_class, categorical_features):
    data_orig = StandardDataset(df,
                                   label_name='income-per-year',
                                   favorable_classes=['>50K', '>50K.'],
                                   protected_attribute_names=protected_attribute_name,
                                   privileged_classes=[privileged_class],
                                   categorical_features=categorical_features
                                   #custom_preprocessing=custom_preprocessing
                                   )
    X = data_orig.features
    y = data_orig.labels.ravel()
    return data_orig, X, y

def load_compas_data(df, protected_attribute_name, privileged_class, categorical_features):
    data_orig = StandardDataset(df,
                                   label_name='two_year_recid',
                                   favorable_classes=[0],
                                   protected_attribute_names=protected_attribute_name,  # 'sex'
                                   privileged_classes=[privileged_class],
                                   # features_to_keep=['sex', 'age', 'race', 'score_text', 'priors_count',
                                   #                   'days_b_screening_arrest', 'c_charge_degree', 'decile_score',
                                   #                   'two_year_recid', 'length_of_stay', 'is_recid_no', 'is_recid_yes'],
                                   categorical_features=categorical_features
                                   # custom_preprocessing=custom_preprocessing
                                   )
    X = data_orig.features
    y = data_orig.labels.ravel()
    return data_orig, X, y

def load_bank_data(df, protected_attribute_name, privileged_class, categorical_features):
    data_orig = StandardDataset(df,
                                   label_name='y',
                                   favorable_classes=['yes'],
                                   protected_attribute_names=protected_attribute_name,  # 'sex'
                                   privileged_classes=[privileged_class],
                                   # features_to_keep=[],
                                   categorical_features=categorical_features
                                   # custom_preprocessing=custom_preprocessing
                                   )
    X = data_orig.features
    y = data_orig.labels.ravel()
    return data_orig, X, y

def load_titanic_data(df, protected_attribute_name, privileged_class, categorical_features):
    data_orig = StandardDataset(df,
                                   label_name='Survived',
                                   favorable_classes=[1],
                                   protected_attribute_names=protected_attribute_name,  # 'sex'
                                   privileged_classes=[privileged_class],
                                   # features_to_keep=[],
                                   categorical_features=categorical_features
                                   # custom_preprocessing=custom_preprocessing
                                   )
    X = data_orig.features
    y = data_orig.labels.ravel()
    return data_orig, X, y
