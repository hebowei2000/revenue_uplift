from dragonnet import DragonNet
import torch
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from utils import results_to_df, auuc_metric, auqc_metric, kendall_metric, lift_h_metric
from sklift import metrics
import optuna
import os.path

road='/data/home/baldwinhe/revenue_uplift/datasets/Hillstrom'# 数据存放的地址

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    
if __name__ == '__main__':
    # Jobs dataset, args: 24, 200, 200.
    # IHDP dataset args: 24, 200 200

    dataset = 'Hillstrom_men'
    model_name = 'dragonnets'
    ranking_lambda = 1.0 #1.0
    
    set_random_seed(8) #Men:9 UR+RR:10; Wom:8/5

    name = f"{dataset}_{model_name}_{ranking_lambda}"
    print(f"Output name: {name}")
    output_dir = '/data/home/baldwinhe/revenue_uplift/dragonnets_ranking/output'
    path = os.path.join(output_dir, name)
    print('path', path)
    
    if not os.path.isdir(path):
        os.makedirs(path)


    train_df =  pd.read_pickle(road + '/dt_binary_men_tv_train.pkl')
    test_df =  pd.read_pickle(road + '/dt_binary_men_test.pkl')
    
    in_features = ['recency', 'history_segment', 'mens', 'womens',
       'zip_code', 'newbie', 'channel_Multichannel', 'channel_Phone', 'channel_Web']
    label_feature = ['spend']
    treatment_feature = ['CASE']
    
    X_train = train_df[in_features].values.astype(float)
    y_train = train_df[label_feature].values.astype(float)
    t_train = train_df[treatment_feature].values.astype(float)

    X_test = test_df[in_features].values.astype(float)
    y_test = test_df[label_feature].values.astype(float)
    t_test = test_df[treatment_feature].values.astype(float)
    
    model = DragonNet(X_train.shape[1], ranking_lambda, epochs=10) #10
    model.fit(X_train, y_train, t_train)
    y0_pred, y1_pred, t_pred, _ = model.predict(X_test)
    
    test_df['target_dif'] = y1_pred - y0_pred
    
    # AUUC (Area under Uplift Curve)
    auuc = auuc_metric(test_df,'target_dif', 100, treatment_feature, label_feature, path)
    
    # lift@30%
    lift_h = lift_h_metric(test_df,'target_dif', 100, treatment_feature, label_feature, h=0.3)
    
    # AUQC (Area under Qini Curve)
    auqc = auqc_metric(test_df,'target_dif', 100, treatment_feature, label_feature, path)
    
    # KRCC (Kendall Rank Correlation Coefficient)
    krcc = kendall_metric(test_df,'target_dif', 100, treatment_feature, label_feature)
    
    print('===========================  Test Results Summary ====================')
    print('auuc', auuc)
    print('lift_h', lift_h)
    print('auqc', auqc)
    print('krcc', krcc)
    