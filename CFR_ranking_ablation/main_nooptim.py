import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
import os.path
from datetime import datetime
from time import time
from torch import nn as nn

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# import optuna
# from optuna.trial import TrialState

import torch
import numpy as np
from config import Config
#from data_preprocessing import IHDP, Jobs, kuaishou, save_normalized_model
from data_preprocessing import Histrom_Binary_Men, Histrom_Binary_Wom, data_to_device
#from evaluate import evaluate
#from lr_model import LogisticRegressionNet
from model import CfrNet
from utils import results_to_df, auuc_metric, auqc_metric, kendall_metric, lift_h_metric
from tqdm import tqdm

# import wandb

# wandb.init(project="my-test-project")

treatment_feature = ['CASE']
outcome_feature = ['spend']
uplift_val = 'target_dif'




def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def train_val_loop(net, train_set, val_set, test_set):
    config = Config()

    kwargs = {'num_workers': 2, 'pin_memory': True} if config.use_gpu else {}
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True, **kwargs)

    train_losses = []
    val_losses = []
    print('start training')
    
    for i in tqdm(range(config.num_epochs)):
        avg_train_loss = 0
        avg_val_loss = 0
        for i, data in enumerate(train_loader):
            if config.use_gpu:
                data = data_to_device(data, config.device)
            
            loss = net.calculate_loss(data, config)

            net.optim.zero_grad()
            loss.backward()
            net.optim.step()

            avg_train_loss += loss
        avg_train_loss = avg_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                if config.use_gpu:
                    data = data_to_device(data, config.device)
            
            loss = net.calculate_loss(data, config)
            avg_val_loss += loss
            avg_val_loss = avg_val_loss / len(val_loader)
            
            val_losses.append(avg_val_loss)
    
    with torch.no_grad():   
        predict_y1, predict_y0 = net.predict(test_set.features.to(config.device))
        test_set.data['target_dif'] = predict_y1.cpu().numpy() - predict_y0.cpu().numpy()
            
    """"     
    iter = range(config.num_epochs)
    if config.do_log_epochs:
        iter = tqdm(iter)

    for epoch in iter:
        train_loss = train(train_loader, net, config)
        train_losses.append(train_loss)
        val_loss, treatment, predict_y1, predict_y0, label = val(val_loader, net, config)
        val_losses.append(val_loss)
        if config.do_log_epochs and epoch % 1 == 0:
            print(f"Epoch {epoch+1}, train loss: {train_loss}, val loss: {val_loss}")
        
    """    
        
    return train_losses, val_losses, predict_y1, predict_y0, test_set.data



if __name__ == '__main__':
    # Jobs dataset, args: 24, 200, 200.
    # IHDP dataset args: 24, 200 200
    
    now = datetime.now()
    str_date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
    config = Config()
    set_random_seed(config.random_seed)
    if not os.path.isdir(config.output_dir) and config.do_save:
        os.makedirs(config.output_dir)


    all_results = []

    
    dataset = config.dataset
    model_name = config.model_name

    if model_name == "cfrnet":
        ipm_function = f"{config.ipm_function}_"
    else:
        ipm_function = ""
    name = f"{dataset}_{model_name}_{ipm_function}{str_date_time}"
    print(f"Output name: {name}")
    path = os.path.join(config.output_dir, name)
    print('path', path)
    
    if not os.path.isdir(path) and config.do_save:
        os.makedirs(path)

    if dataset == "ihdp":
        train_set = IHDP(i, "train")
        val_set = IHDP(i, "val")
    elif dataset == "jobs":
        train_set = Jobs(i, "train")
        val_set = Jobs(i, "val")
    elif dataset == "histrom_binary_men":
        train_set = Histrom_Binary_Men('train')
        val_set = Histrom_Binary_Men('val')
        test_set = Histrom_Binary_Men('test')
    elif dataset == "histrom_binary_wom":
        train_set = Histrom_Binary_Wom('train')
        val_set = Histrom_Binary_Wom('val')
        test_set = Histrom_Binary_Wom('test')
        

    print('data loaded ')
    # Create model
    if model_name == "cfrnet":
        net = CfrNet(train_set.features.shape[1], config.hidden_dim_rep, config.hidden_dim_hypo, config).cuda()
        # net = nn.DataParallel(net, device_ids=device_ids)
    elif model_name == "tarnet":
        net = CfrNet(train_set.features.shape[1], config.hidden_dim_rep, config.hidden_dim_hypo, config, use_ipm=False).cuda()
        # net = nn.DataParallel(net, device_ids=device_ids)


    start = time()
    
    train_losses, val_losses, predict_y1, predict_y0, test_df = train_val_loop(net, train_set, val_set, test_set)
    print(f"Time for realization: {time() - start:.2f}s\nFinal val loss: {val_losses[-1]}")
    # Optional
    # wandb.watch(net)
    

    
    if config.save_main_model == True:
        torch.save(net.state_dict(), 'saved_model/' + model_name + '_all.pt')
    #if config.save_normalized_model == True:
    #    save_normalized_model('saved_model/ss_normal_all.pkl')

    # AUUC (Area under Uplift Curve)
    auuc = auuc_metric(test_df,'target_dif', 100, treatment_feature, outcome_feature, path)
    
    # lift@30%
    lift_h = lift_h_metric(test_df,'target_dif', 100, treatment_feature, outcome_feature, h=0.3)
    
    # AUQC (Area under Qini Curve)
    auqc = auqc_metric(test_df,'target_dif', 100, treatment_feature, outcome_feature, path)
    
    # KRCC (Kendall Rank Correlation Coefficient)
    krcc = kendall_metric(test_df,'target_dif', 100, treatment_feature, outcome_feature)
    
    print('===========================  Test Results Summary ====================')
    print('auuc', auuc)
    print('lift_h', lift_h)
    print('auqc', auqc)
    print('krcc', krcc)
