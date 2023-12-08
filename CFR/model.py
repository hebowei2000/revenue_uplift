import torch
import torch.nn as nn

from discrepancy import maximum_mean_discrepancy_loss, wasserstein_distance
from utils import get_data_with_treatment_type
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

        self.rep = Net(in_dim, in_dim, hidden_dim_rep, config)
        self.h1 = Net(in_dim, 1, hidden_dim_hypo, config) #for t = 1
        self.h0 = Net(in_dim, 1, hidden_dim_hypo, config) # for t = 0
        self.h = Net(in_dim, 2, hidden_dim_hypo, config)

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
            outputs_h1 = outputs_h[treatment_type.squeeze(), 0].unsqueeze(dim=-1)
            outputs_h0 = outputs_h[(~treatment_type).squeeze(), 1].unsqueeze(dim=-1)

        # Calculate prediction loss
        pred_loss = self.calculate_prediction_loss(
            (outputs_h1, outputs_h0),
            (treated_y, control_y),
            (treated_weights, control_weights)
        )

        ##prediction loss + rank loss

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
        
        #final loss
        total_loss = pred_loss + ipm_loss
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
            y_h1 = y_h[:, 0]
            y_h0 = y_h[:, 1]
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
        loss = (treated_weights * self.criterion(outputs_h1, treated_y)).mean()
        loss += (control_weights * self.criterion(outputs_h0, control_y)).mean()
        
        return loss