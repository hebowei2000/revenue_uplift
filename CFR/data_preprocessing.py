import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
road='/data/home/baldwinhe/revenue_uplift/datasets/Hillstrom'# 数据存放的地址


class Histrom_Binary_Men_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.in_features = ['recency', 'history_segment', 'mens', 'womens', 'zip_code', 'newbie', 'channel_Multichannel', 'channel_Phone', 'channel_Web']
        self.label_feature = ['spend']
        self.treatment_feature = ['CASE']
        
        self.features = torch.Tensor(self.data[self.in_features].values.astype(float))
        self.treatment_type = torch.Tensor(self.data[self.treatment_feature].values.astype(float))
        self.labels = torch.Tensor(self.data[self.label_feature].values.astype(float))
        self.u = self.treatment_type.mean()
        self.weights = self.treatment_type / (2*self.u) + (1 - self.treatment_type) / (2*(1-self.u))
         
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.features[index], self.treatment_type[index], self.labels[index], self.weights[index]
        
def Histrom_Binary_Men(set):
    if set == 'train':
        dataset = pd.read_pickle(road + '/dt_binary_men_tv_train.pkl')
        return Histrom_Binary_Men_Dataset(dataset)
        
    elif set == 'val':
        dataset = pd.read_pickle(road + '/dt_binary_men_tv_val.pkl')
        return Histrom_Binary_Men_Dataset(dataset)
    
    elif set == 'test':
        dataset = pd.read_pickle(road + '/dt_binary_men_test.pkl')
        return Histrom_Binary_Men_Dataset(dataset)
    
def data_to_device(data, device):
    for i in range(len(data)):
        data[i] = data[i].to(device)
    return data
        
        