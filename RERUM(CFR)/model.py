import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from discrepancy import maximum_mean_discrepancy_loss, wasserstein_distance
from ziln import zero_inflated_lognormal_pred, zero_inflated_lognormal_loss
from utils import get_data_with_treatment_type


def bmc_loss(pred, target, noise_var):
    logits = - (pred - target.T).pow(2) / (2 * noise_var)
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(pred.device))
    loss = loss * (2 * noise_var).detach()
    
    return loss
    
class BMCLoss(_Loss):
    def  __init__(self, init_noise_sigma, config):
        self.config = config
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma)).to(self.config.device)     
        print('self.noise_sigma.data', self.noise_sigma.data)  
         
    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        noise_var.to(self.config.device)
        return bmc_loss(pred, target, noise_var)    
        
        
class Net(nn.Module):
    def __init__(self, in_dim, out_dmin, hidden_dmin, config):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dmin),
            nn.ELU(),
            nn.Linear(hidden_dmin, hidden_dmin),
            nn.ELU(),
            nn.Linear(hidden_dmin, out_dmin),
        )

        if config.use_gpu:
            self.net.to(config.device)

    def forward(self, x):
        return self.net.forward(x)


class CfrNet(nn.Module):
    def __init__(self, in_dim, hidden_dim_rep, hidden_dim_hypo, config, use_ipm=True):
        super(CfrNet, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.use_ipm = use_ipm
        self.lambda_uplift_ranking = 0.01 # mmd+Men: 0.01 / wass+men: 0.001 / wass+wom: 0.001
        self.lambda_outcome_ranking = 0.001 / (64**2) # Men: 0.001, matrix: 0.001 / (64**2)  // wass+wom: 0.005 / (64**2)

        self.rep = Net(in_dim, in_dim, hidden_dim_rep, config)
        self.h1 = Net(in_dim, 3, hidden_dim_hypo, config) #for t = 1
        self.h0 = Net(in_dim, 3, hidden_dim_hypo, config) # for t = 0
        self.h = Net(in_dim, 6, hidden_dim_hypo, config)

        self.split_h = config.split_h
        if config.split_h:
            h_params = list(self.h1.parameters()) + list(self.h0.parameters())
        else:
            h_params = list(self.h.parameters())
        all_params = list(self.rep.parameters()) + h_params
        self.optim = torch.optim.Adam([
            {'params': self.rep.parameters()},
            {'params': h_params, 'weight_decay': config.weight_decay} # Use regularization for h network
        ], lr=config.learning_rate)
        # self.optim = torch.optim.Adam(all_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    def calculate_loss(self, data, config):
        features, treatment_type, labels, weights = data

        treated_y, control_y = get_data_with_treatment_type(labels, treatment_type)
        treated_weights, control_weights = get_data_with_treatment_type(weights, treatment_type)
        # use features to get the representation of the feature embedding
        representation_output = self.rep(features)
        rep_treated, rep_control = get_data_with_treatment_type(representation_output, treatment_type)

        if self.split_h:
            #split treatment group and control group
            outputs_h1 = self.h1(rep_treated)
            outputs_h0 = self.h0(rep_control)
        else:
            outputs_h = self.h(representation_output)

            # The first 'head' is modeled as the prediction for the treated group
            # The second 'head' is modeled as the prediction for the control group
            outputs_h1 = outputs_h[treatment_type.squeeze(), :3]
            outputs_h0 = outputs_h[(~treatment_type).squeeze(), 3:]

        
        # counterfactual prediction
        if self.split_h:
            # split treatment group and control group
            output_h1_counterfactual = self.h0(rep_treated)
            output_h0_counterfactual = self.h1(rep_control)
        else:
            output_h1_counterfactual = outputs_h[treatment_type.squeeze(), 3:]
            output_h0_counterfactual = outputs_h[(~treatment_type).squeeze(), :3]
       

        # Calculate prediction loss
        pred_loss = self.calculate_prediction_loss(
            (outputs_h1, outputs_h0),
            (treated_y, control_y),
            (treated_weights, control_weights)
        )
        
        ## Calculate uplift ranking loss
        uplift_ranking_loss = self.calculate_uplift_ranking_loss(
            (outputs_h1, outputs_h0),
            (output_h1_counterfactual, output_h0_counterfactual),
            (treated_y, control_y)
        )
        
        outcome_ranking_loss = self.calculate_outcome_ranking_loss(
            (outputs_h1, outputs_h0),
            (treated_y, control_y),
            config            
        )
        
        print('pred_loss', pred_loss)
        print('uplift_ranking_loss', uplift_ranking_loss)
        print('outcome_ranking_loss', outcome_ranking_loss)

        # Add IPM loss
        if self.use_ipm:
            if config.ipm_function == "mmd":
                ipm_loss = config.alpha * maximum_mean_discrepancy_loss(rep_treated, rep_control)
            elif config.ipm_function == "wasserstein":
                ipm_loss = config.alpha * wasserstein_distance(rep_treated, rep_control, config)
            else:
                raise Exception(f"Unknown ipm function: {config.ipm_function}")
        else:
            ipm_loss = 0
        
        #print('ipm_loss', ipm_loss)
        #final loss
        total_loss = pred_loss + ipm_loss + self.lambda_uplift_ranking * uplift_ranking_loss + self.lambda_outcome_ranking * outcome_ranking_loss
        #total_loss = pred_loss + ipm_loss + self.lambda_outcome_ranking * outcome_ranking_loss
        #total_loss = pred_loss + ipm_loss + self.lambda_uplift_ranking * uplift_ranking_loss
        #total_loss = pred_loss + ipm_loss
        return total_loss

    def predict(self, features):
        """Predict treated and control y for the features"""
        rep = self.rep(features)
        if self.split_h:
            y_h1 = self.h1(rep)
            y_h0 = self.h0(rep)
        else:
            y_h = self.h(features) 
            # out_dim = multi value treatment (treatment number)
            y_h1 = y_h[:, :3]
            y_h0 = y_h[:, 3:]
        
        y_h1 = zero_inflated_lognormal_pred(y_h1)
        y_h0 = zero_inflated_lognormal_pred(y_h0)
        return y_h1, y_h0

    def calculate_prediction_loss(self, y_pred, y, weights):
        """
        Calculate the prediction loss.
        y_pred is a tuple (treated, control)
        Same holds for y and weights
        """
        outputs_h1, outputs_h0 = y_pred
        treated_y, control_y = y
        treated_weights, control_weights = weights
        #loss = (treated_weights * self.criterion(outputs_h1, treated_y)).mean()
        #loss += (control_weights * self.criterion(outputs_h0, control_y)).mean()
        
        loss = (treated_weights * zero_inflated_lognormal_loss(treated_y, outputs_h1)).mean()
        loss += (control_weights * zero_inflated_lognormal_loss(control_y, outputs_h0)).mean()
        return loss

    
    
    def calculate_uplift_ranking_loss(self, y_pred, y_counterfactual, y):
        # listwise ranking loss for uplift
        outputs_h1, outputs_h0 = y_pred
        outputs_h1_counterfactual, outputs_h0_counterfactual = y_counterfactual
        outputs_h1 = zero_inflated_lognormal_pred(outputs_h1)
        outputs_h0 = zero_inflated_lognormal_pred(outputs_h0)
        outputs_h1_counterfactual = zero_inflated_lognormal_pred(outputs_h1_counterfactual)
        outputs_h0_counterfactual = zero_inflated_lognormal_pred(outputs_h0_counterfactual)
        
        
        tau_h1 = outputs_h1 - outputs_h1_counterfactual
        tau_h0 = outputs_h0_counterfactual - outputs_h0
        softmax_tau_h1 = F.softmax(tau_h1, dim=0)
        softmax_tau_h0 = F.softmax(tau_h0, dim=0)
        #softmax_tau_h1 = F.softmax(torch.concat([tau_h1, tau_h0], dim=0), dim=0)[:tau_h1.shape[0]]
        #softmax_tau_h0 = F.softmax(torch.concat([tau_h1, tau_h0], dim=0), dim=0)[tau_h1.shape[0]:]
        treated_y, control_y = y
        N1 = outputs_h1.shape[0]
        N0 = outputs_h0.shape[0]
        loss = - (N1 + N0) * ((1/N1)*torch.sum(treated_y * torch.log(softmax_tau_h1)) - (1/N0)*torch.sum(control_y * torch.log(softmax_tau_h0)))
        return loss
    
    def calculate_outcome_ranking_loss(self, y_pred, y, config):
        # enchance the rankability of outcome regression inner the treatment/control group 
        outputs_h1, outputs_h0 = y_pred
        treated_y, control_y = y
        outputs_h1 = zero_inflated_lognormal_pred(outputs_h1)
        outputs_h0 = zero_inflated_lognormal_pred(outputs_h0)
        
        # pairwise outcome ranking loss in treatment group
        """
        treat_loss = torch.tensor([0.0]).to(config.device)
        for i in torch.randint(low=0,high=outputs_h1.shape[0],size=(100,)):
            for j in torch.randint(low=0,high=outputs_h1.shape[0],size=(100,)):
                pair_loss = ((outputs_h1[i] - outputs_h1[j]) - (treated_y[i] - treated_y[j])) ** 2 if (outputs_h1[i] - outputs_h1[j]) * (treated_y[i] - treated_y[j]) < 0 else torch.tensor([0.0]).to(config.device)
                treat_loss += pair_loss
        """    
        
        outputs_h1_matrix = outputs_h1 - outputs_h1.T
        treated_y_matrix = treated_y - treated_y.T
        product = outputs_h1_matrix * treated_y_matrix
        new_tensor = torch.zeros_like(outputs_h1_matrix).to(config.device)
        mask = product >= 0
        new_tensor = (outputs_h1_matrix - treated_y_matrix) ** 2
        new_tensor[mask] = 0.0
        treat_loss = torch.sum(new_tensor)
        
           
        
        # pairwise outcome ranking loss in control group
        """
        control_loss = torch.tensor([0.0]).to(config.device)
        for i in torch.randint(low=0,high=outputs_h0.shape[0],size=(100,)):
            for j in torch.randint(low=0,high=outputs_h0.shape[0],size=(100,)):
                pair_loss = ((outputs_h0[i] - outputs_h0[j]) - (control_y[i] - control_y[j])) ** 2 if (outputs_h0[i] - outputs_h0[j]) * (control_y[i] - control_y[j]) < 0 else torch.tensor([0.0]).to(config.device)
                control_loss += pair_loss
        """    
               
        outputs_h0_matrix = outputs_h0 - outputs_h0.T
        control_y_matrix = control_y - control_y.T
        product = outputs_h0_matrix * control_y_matrix
        new_tensor = torch.zeros_like(outputs_h0_matrix).to(config.device)
        mask = product >= 0
        new_tensor = (outputs_h0_matrix - control_y_matrix) ** 2
        new_tensor[mask] = 0.0
        control_loss = torch.sum(new_tensor)
        
            
        return treat_loss + control_loss
        
        
    