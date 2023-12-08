import torch
import numpy as np 
from tqdm import tqdm
from model import CfrNet
from config import Config

def train_and_val(train_loader, val_loader, net: CfrNet, config: Config):
    kwargs = {'num_workers': 2, 'pin_memory': True} if config.use_gpu else {}
    train_losses = []
    val_losses = []
    for i in tqdm(range(config.epoch)):
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





