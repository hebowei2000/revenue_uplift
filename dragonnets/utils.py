from typing import List, Iterable
import pandas as pd
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

def error_ATT(pred_y_t, pred_y_c, test_set):
    t = test_set.t
    y = test_set.y
    e = test_set.e
    """
    Calculates the                                                                                                                                                                                                                                                                    on the average treatment effect.
    :param pred_y_c: predicted y value for control
    :param pred_y_t: predicted y value for treated
    :param t: indication if sample was treated or not
    :param y: outcome of samples
    :param e: indication if sample was randomized
    :return: error on ATT value
    """

    # True ATT value
    # t = 0 and e = 1 indicates sample from C intersect E
    ATT = np.mean(y[t > 0]) - np.mean(y[(1 - t + e) > 1])

    # Predicted individual treatment effect
    ITE_pred = pred_y_t - pred_y_c

    # Only consider prediction on samples that were treated?
    # Their implementation takes (t + e > 1)
    ATT_pred = ITE_pred[(t > 0)].mean()

    err_ATT = abs(ATT - ATT_pred).item()

    return err_ATT


def ite(test_set):
    """
    Calculate the true Individual Treatment Effect (ITE) for all elements in the test set
    """
    pass


def error_ATE(y_h1, y_h0, test_set):
    """Calculate the error in the average treatment effect (ATE)
    We first compute the actual ATE.
    For this we construct the treated and control set, with the same size as the test set.
    We use the ycf field of the dataset to do so.
    First we build the treated set by concatenating the y values of the patients that were treated and the counterfactual
        y values of the patients that were not treated
    Second we build the control set by concatenating the y values of the patients that were not treated and the counterfactual
        y values of the patients that were treated
    The true ATE is then the mean of the differences between the treated and control sets
    The predicted ATE is the mean of the differences between the output of the h1 network and the h0 network.
    The error is their absolute difference
    """
    actual_treated_y = test_set.y[test_set.t]
    cf_treated_y = test_set.ycf[~test_set.t]
    treated = np.concatenate((actual_treated_y, cf_treated_y))

    actual_control_y = test_set.y[~test_set.t]
    cf_control_y = test_set.ycf[test_set.t]
    control = np.concatenate((cf_control_y, actual_control_y))

    ate_actual = (treated - control).mean()
    ate_pred = (y_h1 - y_h0).mean()

    return torch.abs(ate_actual - ate_pred).item()


def error_PEHE(y_h1, y_h0, test_set):
    actual_treated_y = test_set.y[test_set.t]
    cf_treated_y = test_set.ycf[~test_set.t]
    treated = np.concatenate((actual_treated_y, cf_treated_y))

    actual_control_y = test_set.y[~test_set.t]
    cf_control_y = test_set.ycf[test_set.t]
    control = np.concatenate((cf_control_y, actual_control_y))

    return np.sqrt(np.square((treated - control) - (y_h1.numpy() - y_h0.numpy())).mean())


def R_pol(pred_y_t, pred_y_c, test_set, t, y):
    t = test_set.t
    y = test_set.y
    """
    Calculates the average value loss when treating with a policy based on ITE.
    :param prediction: tensor of size (n,2): predicted y value for treated and non-treated
    :param t: indication if sample was treated or not
    :param y: outcome of samples
    :param e: indication if sample was randomized
    :return: policy risk
    """
    # ITE based on predictions
    ITE_pred = pred_y_t - pred_y_c

    # Treat if predicted ITE > lambda. Table 1 takes lambda = 0.
    lam = 0
    policy = (ITE_pred > lam).numpy()

    # Expectations of Y_0 and Y_1 given policy and t
    avg_treat_value = (y[(policy == t) * (t > 0)]).sum() / len(y)
    avg_control_value = (y[(policy == t) * (t < 1)]).sum() / len(y)

    # Probability of treating
    p = policy.mean() 

    # Estimation of the policy risk
    policy_risk = 1 - p * avg_treat_value - (1 - p) * avg_control_value

    return policy_risk.item()


def get_data_with_treatment_type(data, treatment):
    treatment = treatment.squeeze()
    treated = data[treatment == 1]
    control = data[treatment == 0]
    return treated, control


def get_computing_device(use_gpu=False):
    return torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")


def data_to_device(data: Iterable[torch.Tensor], device) -> Iterable[torch.Tensor]:
    return (x.to(device) for x in data)


def results_to_df(all_results):
    keys = list(all_results[0].keys())
    new_dict = {k: [] for k in keys}
    for res in all_results:
        for k in keys:
            new_dict[k].append(res[k])

    return pd.DataFrame(new_dict)


def auuc_metric(data, uplift_val, bucket_num, treatment_feature, label_feature, path, final_test = False):
    if final_test:
        del data
    data = data.sort_values(by=uplift_val, ascending = False)
    print(data.shape)
    print('data bucket')
    data['bucket'] = pd.qcut(-data[uplift_val], bucket_num, labels=False, duplicates='drop')
    print('====计算完毕====')
    print('====生成随机数====')
    data = data.sort_values(by=treatment_feature)
    l0, l1 = data.loc[data[treatment_feature[0]]==0], data.loc[data[treatment_feature[0]]==1]
    mean0, mean1 = l0[label_feature].values.mean(), l1[label_feature].values.mean()
    print('treatment = 0:', mean0, 'treatment = 1:', mean1)
    r0, r1 = (np.random.rand(len(l0)) - 0.5)/100000 + mean0, (np.random.rand(len(l1)) - 0.5)/100000 + mean1
    data['random'] = list(r0) + list(r1)
    bucket_diff = []
    print('=========')
    for i in set(data.bucket):
        bucket_diff.append(np.mean(data.loc[(data.bucket == i)&(data[treatment_feature[0]] == 1), label_feature].values)\
                        /np.mean(data.loc[(data.bucket == i)&(data[treatment_feature[0]] == 0), label_feature].values)-1)
    
    res, population, rand_res, bucket_len = [], [], [], []
    for i in np.arange(0, len(set(data.bucket)), 1):
        dbucket = data.loc[data.bucket <= i]
        db_base = dbucket.loc[dbucket[treatment_feature[0]] == 0]
        db_exp = dbucket.loc[dbucket[treatment_feature[0]] == 1]
        #cumugain = (db_exp[label_feature].mean()/db_base[label_feature].mean()-1) * (len(db_base) + len(db_exp))
        cumugain = (db_exp[label_feature].mean() - db_base[label_feature].mean()) * (len(db_base) + len(db_exp))
        #cumu_random = (db_exp['random'].mean()/db_base['random'].mean()-1) * (len(db_base) + len(db_exp))
        cumu_random = (db_exp['random'].mean() - db_base['random'].mean()) * (len(db_base) + len(db_exp))
        population.append(len(db_base) + len(db_exp))
        bucket_len.append(len(data[data.bucket == i]))
        res.append(cumugain)
        rand_res.append(cumu_random)
    rand_res[-1] = res[-1]
    cumuGain = pd.DataFrame({'cumuGain': res,
                             'population': population,
                             'percent': np.arange(1 / len(set(data.bucket)), 1 + 1 / len(set(data.bucket)),
                                                  1 / len(set(data.bucket))),
                             'random': rand_res})
    
    # 归一化 将值框定在[0，1]之间
    cumugain = cumuGain['cumuGain']
    gap0 = cumugain.iloc[-1].values
    cumugain = [(i) / abs(gap0) for i in cumugain]

    cumu_random = cumuGain['random']
    gap = cumu_random.iloc[-1]
    cumu_random = [(i) / abs(gap) for i in cumu_random]

    plt.plot(np.append(0,(np.array(population))/max(population)), np.append(0,cumugain), marker='*', label='pred')
    plt.plot(np.append(0,(np.array(population))/max(population)), np.append(0,cumu_random), marker='*', label='random')
    plt.grid(linestyle='--')
    plt.xlabel('Cumulative percentage of people targeted')
    plt.ylabel('Cumulative uplift')
    plt.legend()
    plt.savefig(path + '/Uplift Curves.png' )
        
    # 近似计算cumugain函数和x轴围成的曲线下面积，也就是auuc
    auuc_value, rand = np.trapz(np.append(0, cumugain),np.append(0, (np.array(population))/max(population))),\
                                np.trapz(np.append(0, cumu_random),np.append(0, (np.array(population))/max(population)))

    print('raw auuc score', auuc_value)    
    spearmanr_value = scipy.stats.spearmanr(bucket_diff, list(set(data.bucket)))[0]
    
    print(cumugain, np.array(population)/max(population))
    if gap0 < 0: 
        auuc_value, rand = np.trapz([i + 1 for i in np.append(0, cumugain)], np.append(0,(np.array(population))/max(population))),\
                                    np.trapz([i + 1 for i in np.append(0, cumu_random)], np.append(0,(np.array(population))/max(population)))
    print('auuc score:', auuc_value, 'random score:', rand)
    print('spearmanr:', spearmanr_value)
    return auuc_value


def auqc_metric(data, uplift_val, bucket_num, treatment_feature, label_feature, path, final_test = False):
    if final_test:
        del data
    data = data.sort_values(by=uplift_val, ascending = False)
    print(data.shape)
    print('data bucket')
    data['bucket'] = pd.qcut(-data[uplift_val], bucket_num, labels=False, duplicates='drop')
    print('====计算完毕====')
    print('====生成随机数====')
    data = data.sort_values(by=treatment_feature)
    l0, l1 = data.loc[data[treatment_feature[0]]==0], data.loc[data[treatment_feature[0]]==1]
    mean0, mean1 = l0[label_feature].values.mean(), l1[label_feature].values.mean()
    print('treatment = 0:', mean0, 'treatment = 1:', mean1)
    r0, r1 = (np.random.rand(len(l0)) - 0.5)/100000 + mean0, (np.random.rand(len(l1)) - 0.5)/100000 + mean1
    data['random'] = list(r0) + list(r1)
    
    res, population, rand_res, bucket_len = [], [], [], []
    for i in np.arange(0, len(set(data.bucket)), 1):
        dbucket = data.loc[data.bucket <= i]
        db_base = dbucket.loc[dbucket[treatment_feature[0]] == 0]
        db_exp = dbucket.loc[dbucket[treatment_feature[0]] == 1]
        cumugain = db_exp[label_feature].sum() - db_base[label_feature].sum() * (len(db_exp) / len(db_base))
        cumu_random = db_exp['random'].sum() - db_base['random'].sum() * (len(db_exp) / len(db_base))
        population.append(len(db_base) + len(db_exp))
        bucket_len.append(len(data[data.bucket == i]))
        res.append(cumugain)
        rand_res.append(cumu_random)
    rand_res[-1] = res[-1]
    cumuGain = pd.DataFrame({'cumuGain': res,
                             'population': population,
                             'percent': np.arange(1 / len(set(data.bucket)), 1 + 1 / len(set(data.bucket)),
                                                  1 / len(set(data.bucket))),
                             'random': rand_res})
    
    # 归一化 将值框定在[0，1]之间
    cumugain = cumuGain['cumuGain']
    gap0 = cumugain.iloc[-1].values
    cumugain = [(i) / abs(gap0) for i in cumugain]

    cumu_random = cumuGain['random']
    gap = cumu_random.iloc[-1]
    cumu_random = [(i) / abs(gap) for i in cumu_random]

    plt.plot(np.append(0,(np.array(population))/max(population)), np.append(0,cumugain), marker='*', label='pred')
    plt.plot(np.append(0,(np.array(population))/max(population)), np.append(0,cumu_random), marker='*', label='random')
    plt.grid(linestyle='--')
    plt.xlabel('Cumulative percentage of people targeted')
    plt.ylabel('Cumulative uplift')
    plt.legend()
    plt.savefig(path + '/Qini Curves.png' )
        
    # 近似计算cumugain函数和x轴围成的曲线下面积，也就是auqc
    auqc_value, rand = np.trapz(np.append(0, cumugain),np.append(0, (np.array(population))/max(population))),\
                                np.trapz(np.append(0, cumu_random),np.append(0, (np.array(population))/max(population)))

    print('raw auqc score', auqc_value)    
    
    print(cumugain, np.array(population)/max(population))
    if gap0 < 0: 
        auqc_value, rand = np.trapz([i + 1 for i in np.append(0, cumugain)], np.append(0,(np.array(population))/max(population))),\
                                    np.trapz([i + 1 for i in np.append(0, cumu_random)], np.append(0,(np.array(population))/max(population)))
    print('auqc score:', auqc_value, 'random score:', rand)
    return auqc_value


def lift_h_metric(data, uplift_val, bucket_num, treatment_feature, label_feature, h=0.3, final_test = False):
    if final_test:
        del data
    data = data.sort_values(by=uplift_val, ascending = False)
    print(data.shape)
    data['bucket'] = pd.qcut(-data[uplift_val], bucket_num, labels=False, duplicates='drop')
    data = data.sort_values(by=treatment_feature)    
    dbucket = data.loc[data.bucket <= h*len(set(data.bucket))]
    db_base = dbucket.loc[dbucket[treatment_feature[0]] == 0]
    db_exp = dbucket.loc[dbucket[treatment_feature[0]] == 1]
    lift_h_value = db_exp[label_feature].mean() - db_base[label_feature].mean()
    return lift_h_value


def kendall_metric(data, uplift_val, bucket_num, treatment_feature, label_feature, final_test = False):
    if final_test:
        del data          
    data = data.sort_values(by=uplift_val, ascending = False)
    data['bucket'] = pd.qcut(-data[uplift_val], bucket_num, labels=False, duplicates='drop')
    data = data.sort_values(by=treatment_feature)    
    cate_list = []
    pred_uplift_list = []
    for i in np.arange(0, len(set(data.bucket)), 1):
        dbucket = data.loc[data.bucket == i]
        cate = dbucket.loc[dbucket[treatment_feature[0]] == 1][label_feature].mean() - dbucket.loc[dbucket[treatment_feature[0]] == 0][label_feature].mean()
        cate_list.append(cate[0])
        pred_uplift = dbucket[uplift_val].mean()
        pred_uplift_list.append(pred_uplift)
     
    pred_uplift_list_rank = np.argsort(pred_uplift_list)
    cate_list_rank = np.argsort(cate_list)
    print('pred_uplift_list_rank', pred_uplift_list_rank)
    print('cate_list_rank', cate_list_rank)
    correlation, _ = kendalltau(pred_uplift_list_rank, cate_list_rank)
    

    return correlation