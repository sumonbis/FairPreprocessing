import pandas as pd
import csv
import numpy as np
from aif360.metrics import *
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import collections

def fair_metrics(fname, dataset, pred, pred_is_dataset=False):
    filename = fname
    if pred_is_dataset:
        dataset_pred = pred
    else:
        dataset_pred = dataset.copy()
        dataset_pred.labels = pred

    cols = ['Acc', 'F1', 'DI','SPD', 'EOD', 'AOD', 'ERD', 'CNT', 'TI']
    obj_fairness = [[1,1,1,0,0,0,0,1,0]]

    fair_metrics = pd.DataFrame(columns=cols) #data=obj_fairness, index=['objective']

    for attr in dataset_pred.protected_attribute_names:
        idx = dataset_pred.protected_attribute_names.index(attr)
        privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}]
        unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}]

        classified_metric = ClassificationMetric(dataset,
                                                     dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        distortion_metric = SampleDistortionMetric(dataset,
                                                     dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        acc = classified_metric.accuracy()
        f1_sc = 2 * (classified_metric.precision() * classified_metric.recall()) / (classified_metric.precision() + classified_metric.recall())

        mt = [acc, f1_sc,
                        classified_metric.disparate_impact(),
                        classified_metric.mean_difference(),
                        classified_metric.equal_opportunity_difference(),
                        classified_metric.average_odds_difference(),
                        classified_metric.error_rate_difference(),
                        metric_pred.consistency(),
                        classified_metric.theil_index()
                    ]
        w_row = []
        # Convert metrics to logarithmic scale which have a target 1.0
        mt[2] = np.log(mt[2])
        mt[7] = np.log(mt[7])

        # mt = ['%.4f' % elem for elem in mt]
        mt = list(np.around(np.array(mt), 3))

        for i in mt:
            # print("%.8f"%i)
            w_row.append(i) # "%.4f"%i


        # with open(filename, 'a') as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerow(w_row)


        row = pd.DataFrame([mt], columns = cols, index = [attr])
        fair_metrics = fair_metrics.append(row)
    fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)

    ###### Compute new fairness metrics
    # metric_dict = classified_metric.performance_measures()
    #print(metric_dict['TPR'])
    # item_count = classified_metric.num_instances()
    # print(classified_metric.num_negatives())
    # print(classified_metric.num_pred_negatives())
    # print(classified_metric.num_instances())

    # print(fair_metrics)
    return fair_metrics

def count_mismatch(y1_pred, y2_pred):
    mismatch = []
    y1 = y1_pred.tolist()
    y2 = y2_pred.tolist()
    if(len(y1) != len(y2)):
        raise Exception("Test data size do not match.")
    for i in range(len(y1)):
        mismatch.append(y1[i] == y2[i])
    mismatch_count = mismatch.count(False)
    # mismatch = y1_pred != y2_pred
    # mismatch_count = collections.Counter(mismatch)[True]
    return mismatch_count

def get_unprivileged(data):
    for attr in data.protected_attribute_names:
        idx = data.protected_attribute_names.index(attr)
        privileged_groups = [{attr: data.privileged_protected_attributes[idx][0]}]
        unprivileged_groups = [{attr: data.unprivileged_protected_attributes[idx][0]}]
    # privileged_group = privileged_groups[0]
    unprivileged_group = unprivileged_groups[0]
    attr = list(unprivileged_group.keys())[0]
    val = unprivileged_group.get(attr)
    return (attr, val)

def compute_CV(data_orig_test, y1_pred, y2_pred):
    unpriv, unpriv_val = get_unprivileged(data_orig_test)
    print("Unprinv:", unpriv, unpriv_val)

    test_data, _ = data_orig_test.convert_to_dataframe()
    y_test = data_orig_test.labels.ravel()

    mismatch_count = count_mismatch(y1_pred, y2_pred)
    CVR = mismatch_count/len(y_test)
    # print('CVR=', CVR)

    u = 0
    p = 0
    m_u = 0
    m_p = 0

    for i in range(len(y_test)):

        if(test_data[unpriv][i] == unpriv_val): # unprivileged: female
            u += 1
            if(y1_pred[i] != y2_pred[i]):
                m_u += 1
        else:
            p += 1
            if(y1_pred[i] != y2_pred[i]):
                m_p += 1

    CVR_u = m_u/u
    CVR_p = m_p/p
    CVD = CVR_u - CVR_p
    # print('CVD=', CVD, 'CVR_u=', CVR_u, 'CVR_p=', CVR_p)
    return (CVR, CVD)

def compute_new_metrics(data_orig_test, y1_pred, y2_pred):
    CVR, CVD = compute_CV(data_orig_test, y1_pred, y2_pred)
    AVR, AVD = compute_AV(data_orig_test, y1_pred, y2_pred)
    AVR_SPD, AVD_SPD = compute_AV_SPD(data_orig_test, y1_pred, y2_pred)
    AVD_AOD = compute_AV_AOD(data_orig_test, y1_pred, y2_pred)
    AV_ERD = compute_AV_ERD(data_orig_test, y1_pred, y2_pred)
    return CVR, CVD, AVR, AVD, AVR_SPD, AVD_SPD, AVD_AOD, AV_ERD

def compute_AV(data_orig_test, y1_pred, y2_pred):
    unpriv, unpriv_val = get_unprivileged(data_orig_test)
    # print("Unprinv:", unpriv, unpriv_val)

    test_data, _ = data_orig_test.convert_to_dataframe()
    y_test = data_orig_test.labels.ravel()

    favorable = 0
    favorable_p = 0
    favorable_u = 0
    fav_increase_u = 0
    fav_increase_p = 0

    for i in range(len(y_test)):

        if(y_test[i] == 1):
            favorable += 1

            if(test_data[unpriv][i] == unpriv_val): # uprivileged: female

                favorable_u += 1
                if (y1_pred[i] == 0 and y2_pred[i] == 1):
                    fav_increase_u += 1
                if (y1_pred[i] == 1 and y2_pred[i] == 0):
                    fav_increase_u -= 1


            else:
                favorable_p += 1
                if (y1_pred[i] == 0 and y2_pred[i] == 1):
                    fav_increase_p += 1
                if (y1_pred[i] == 1 and y2_pred[i] == 0):
                    fav_increase_p -= 1

        # else:
        #     if(test_data[unpriv][i] == unpriv_val): # uprivileged: female
        #         favorable_u += 1
        #     else:
        #         favorable_p += 1


    fav_increase = fav_increase_u + fav_increase_p
    AVR = fav_increase/favorable # fav_increase/len(y_test)
    AVR_u = fav_increase_u/favorable_u
    AVR_p = fav_increase_p/favorable_p
    AVD = AVR_u - AVR_p
    # print ('AVR =', AVR)
    # print('AVD=', AVD, 'AVR_u=', AVR_u, 'AVR_p=', AVR_p)

    return (AVR, AVD)

def compute_AV_SPD(data_orig_test, y1_pred, y2_pred):
    unpriv, unpriv_val = get_unprivileged(data_orig_test)
    # print("Unprinv:", unpriv, unpriv_val)

    test_data, _ = data_orig_test.convert_to_dataframe()
    y_test = data_orig_test.labels.ravel()

    favorable_p = 0
    favorable_u = 0
    fav_increase_u = 0
    fav_increase_p = 0

    for i in range(len(y_test)):

        if(test_data[unpriv][i] == unpriv_val): # uprivileged: female

            favorable_u += 1
            if (y1_pred[i] == 0 and y2_pred[i] == 1):
                fav_increase_u += 1
            if (y1_pred[i] == 1 and y2_pred[i] == 0):
                fav_increase_u -= 1

        else:

            favorable_p += 1
            if (y1_pred[i] == 0 and y2_pred[i] == 1):
                fav_increase_p += 1
            if (y1_pred[i] == 1 and y2_pred[i] == 0):
                fav_increase_p -= 1

    fav_increase = fav_increase_u + fav_increase_p
    AVR = fav_increase/len(y_test)
    AVR_u = fav_increase_u/favorable_u
    AVR_p = fav_increase_p/favorable_p
    AVD = AVR_u - AVR_p
    # print ('AVR =', AVR)
    # print('AVD=', AVD, 'AVR_u=', AVR_u, 'AVR_p=', AVR_p)

    return (AVR, AVD)

def compute_AV_AOD(data_orig_test, y1_pred, y2_pred):
    AVR_EOD, AVD_EOD = compute_AV(data_orig_test, y1_pred, y2_pred)

    unpriv, unpriv_val = get_unprivileged(data_orig_test)
    # print("Unprinv:", unpriv, unpriv_val)

    test_data, _ = data_orig_test.convert_to_dataframe()
    y_test = data_orig_test.labels.ravel()

    favorable = 0
    favorable_p = 0
    favorable_u = 0
    fav_increase_u = 0
    fav_increase_p = 0

    for i in range(len(y_test)):

        if (y_test[i] == 0):
            favorable += 1

            if (test_data[unpriv][i] == unpriv_val):  # uprivileged: female

                favorable_u += 1
                if (y1_pred[i] == 0 and y2_pred[i] == 1):
                    fav_increase_u += 1
                if (y1_pred[i] == 1 and y2_pred[i] == 0):
                    fav_increase_u -= 1


            else:
                favorable_p += 1
                if (y1_pred[i] == 0 and y2_pred[i] == 1):
                    fav_increase_p += 1
                if (y1_pred[i] == 1 and y2_pred[i] == 0):
                    fav_increase_p -= 1

        # else:
            # if (test_data[unpriv][i] == unpriv_val):  # uprivileged: female
            #     favorable_u += 1
            # else:
            #     favorable_p += 1

    fav_increase = fav_increase_u + fav_increase_p
    AVR = fav_increase / favorable # len(y_test)
    AVR_u = fav_increase_u / favorable_u
    AVR_p = fav_increase_p / favorable_p
    AVD = AVR_u - AVR_p
    AV_AOD = (AVD + AVD_EOD) / 2

    return AV_AOD

def compute_AV_ERD(data_orig_test, y1_pred, y2_pred):

    unpriv, unpriv_val = get_unprivileged(data_orig_test)
    # print("Unprinv:", unpriv, unpriv_val)

    test_data, _ = data_orig_test.convert_to_dataframe()
    y_test = data_orig_test.labels.ravel()

    favorable = 0
    favorable_p = 0
    favorable_u = 0
    fav_increase_u = 0
    fav_increase_p = 0

    for i in range(len(y_test)):

        if (y_test[i] == 0):
            favorable += 1

            if (test_data[unpriv][i] == unpriv_val):  # uprivileged: female

                favorable_u += 1
                if (y1_pred[i] == 0 and y2_pred[i] == 1):
                    fav_increase_u += 1
                if (y1_pred[i] == 1 and y2_pred[i] == 0):
                    fav_increase_u -= 1


            else:
                favorable_p += 1
                if (y1_pred[i] == 0 and y2_pred[i] == 1):
                    fav_increase_p += 1
                if (y1_pred[i] == 1 and y2_pred[i] == 0):
                    fav_increase_p -= 1

        # else:
        #     if (test_data[unpriv][i] == unpriv_val):  # uprivileged: female
        #         favorable_u += 1
        #     else:
        #         favorable_p += 1

    FPR_u = fav_increase_u / favorable_u
    FPR_p = fav_increase_p / favorable_p

    favorable = 0
    favorable_p = 0
    favorable_u = 0
    fav_increase_u = 0
    fav_increase_p = 0

    for i in range(len(y_test)):

        if (y_test[i] == 1):
            favorable += 1

            if (test_data[unpriv][i] == unpriv_val):  # uprivileged: female

                favorable_u += 1
                if (y1_pred[i] == 1 and y2_pred[i] == 0):
                    fav_increase_u += 1
                if (y1_pred[i] == 0 and y2_pred[i] == 1):
                    fav_increase_u -= 1


            else:
                favorable_p += 1
                if (y1_pred[i] == 1 and y2_pred[i] == 0):
                    fav_increase_p += 1
                if (y1_pred[i] == 0 and y2_pred[i] == 1):
                    fav_increase_p -= 1

        # else:
        #     if (test_data[unpriv][i] == unpriv_val):  # uprivileged: female
        #         favorable_u += 1
        #     else:
        #         favorable_p += 1

    FNR_u = fav_increase_u / favorable_u
    FNR_p = fav_increase_p / favorable_p

    ERR_u = FPR_u + FNR_u
    ERR_p = FPR_p + FNR_p

    AV_ERD = ERR_u - ERR_p

    return AV_ERD

def get_fair_metrics_and_plot(fname, data, model, model_aif=False):
    pred = model.predict(data).labels if model_aif else model.predict(data.features)
    fair = fair_metrics(fname, data, pred)
    return (pred, fair)

def get_model_performance(X_test, y_true, y_pred, probs):
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, matrix, f1

def plot_model_performance(model, X_test, y_true):
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    accuracy, matrix, f1 = get_model_performance(X_test, y_true, y_pred, probs)
    print('Accuracy: ' + str(accuracy) + ', F1-score: ' + str(f1))
