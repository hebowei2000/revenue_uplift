import pandas as pd 
import torch


'''
calculate the Maximum Mean Discrepancy loss
'''
def maximum_mean_discrepancy_loss(X_treat, X_control):
    return 2 * torch.norm(X_treat.mean(axis=0) - X_control.mean(axis=0))


"""
    Implementation of algorithm 2 (appendix B.1)
    :param X_treat: mini-batch of treated samples in the form [phi(X),t,y]
    :param X_control: mini-batch of control samples in the form [phi(X),t,y]
    :param t: indication if x is treated or not
    :param p: probability of treated: p = p(t = 1) = sum(t_i) over all i (?)
    :param lamba: smoothing parameter (standard 1)
    :param iterations: ? (standard 10)
    :return: Wasserstein distance between treated and control sample batches
    """

def wasserstein_distance(X_treat, X_control, config, t=None, p=0.5, lamba=1, iterations=10):
    treat_num = X_treat.size(dim=0)
    control_num = X_control.size(dim=0)

    if treat_num==0 or control_num==0:
        return 0

    #compute distance matrix M

   # M = torch.tensor([[torch.linalg.vector_norm(X_treat[i] - X_control[j])**2 for j in range(control_num)] for i in range(treat_num)])
    M = torch.norm(X_treat[:, None] - X_control, dim=2)**2

    #calculate transport matrix T
    a = p * torch.ones((treat_num, 1)).to(config.device) / treat_num
    b = (1 - p) * torch.ones((control_num, 1)).to(config.device) / control_num

    K = torch.exp(-lamba * M)
    K_tilde = K / a
    

    u = a
    for i in range(0, iterations):
        u = 1.0 / torch.matmul(K_tilde, b / torch.matmul(torch.transpose(K, 0, 1), u))
        print('point 3')

    v = b / torch.matmul(torch.transpose(K, 0, 1), u)
    
    print('point 4')
    T = u * (torch.transpose(v, 0, 1) * K)

    # calculate distance
    E = T * M

    return 2 * torch.sum(E)