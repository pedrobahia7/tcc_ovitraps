import torch
import torch.nn as nn
import numpy as np
import pdb
from sklearn.base import BaseEstimator, TransformerMixin

# from utils.NN_building 
# separate NN and create function eval 





class mlp(nn.Module):

    def __init__(self,model_input,model_output,model_type,input_3d):
        super().__init__()
        self.model_type = model_type
        self.input_3d = input_3d    

        if input_3d:
            depth = 4  # Fixed depth as novos + latitutde + longitude + distance
            self.conv1 = nn.Conv1d(in_channels=depth, out_channels=1, kernel_size=depth, padding=1)
            self.flatten_size = model_input 
            self.layer1 = nn.Linear(self.flatten_size, 20) 
        else:
            self.layer1 = nn.Linear(model_input, 10)

        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 5)    
        self.layer4 = nn.Linear(5, model_output)
        self.output_layer = nn.Sigmoid()

    def forward(self, x):
        if self.input_3d:
            x = x.permute(0,2,1)
            x = torch.relu(self.conv1(x))
            x = x.view(x.size(0), -1) 

        out1 = nn.functional.relu(self.layer1(x))
        out2 = nn.functional.relu(self.layer2(out1))
        out3 = nn.functional.relu(self.layer3(out2))
        final_output = self.output_layer(self.layer4(out3))
        return final_output
    
def compute_weights_slow_decay(values, highlight_targets):
        weights = []
        for value in values:
            # Highlight values near the targets
            highlight_factor = sum(1 / (abs(value - target)*0.4 + 1) for target in highlight_targets)
            weights.append(highlight_factor)
        return weights


class MSELoss(nn.Module):
    """
    Class to create a custom loss and calculonge the pdf of its original distribution if necessary. The name of the distribution must 
    be passed in the constructor
    
    Currently available distributions:
    exponential - **params['lamb']

    """
    def __init__(self):
        """
        Distribution: name of the distribution to be used as reference. 
        
        
        """
        super(MSELoss, self).__init__()


    
    def forward(self, y, y_pred):
        #weights = torch.tensor([1 if val < 100 else 100 / val for val in y], dtype=torch.float32)
        weights = compute_weights_slow_decay(y, [19, 35])
        weights = torch.tensor(weights)        
        loss = torch.mean((y - y_pred)**2 * weights)

        return loss
        #return torch.mean((y - y_pred)**2)

        
    """def pdf(self, x, **params):
        if self.model_type == 'exponential_renato':
            lamb = params.get('lamb')
            return lamb*torch.exp(-lamb*x)
        else:
            raise ValueError('Distribution not found')"""

class ExponentialLoss(nn.Module):
    """
    Class to create a custom loss and calculonge the pdf of its original distribution if necessary. The name of the distribution must 
    be passed in the constructor
    
    Currently available distributions:
    exponential - **params['lamb']

    """
    def __init__(self):
        """
        Distribution: name of the distribution to be used as reference. 
        
        
        """
        super(ExponentialLoss, self).__init__()


    def forward(self, y, lamb):

        # Example: NNL of the exponential distribution
        loss = 0
        for i in range(len(lamb)):
            loss += torch.log(lamb[i]) - lamb[i]*y[i]
        return -loss

        
    """def pdf(self, x, **params):
        if self.model_type == 'exponential_renato':
            lamb = params.get('lamb')
            return lamb*torch.exp(-lamb*x)
        else:
            raise ValueError('Distribution not found')"""

class ParetoLoss(nn.Module):
    """
    Class to create a pareto loss 

    """
    def __init__(self):
        """
        Distribution: name of the distribution to be used as reference. 
        
        
        """
        super(ParetoLoss, self).__init__()

    def forward(self, y, alpha, x_m = 1):
        # Example: NNL of the exponential distribution

        alpha = alpha[y>0]
        y = y[y>0] # remove zeros
        loss = 0
        for i in range(y.shape[0]):
            loss +=  torch.log(alpha[i])*x_m - (alpha[i] + 1)*torch.sum(torch.log(y[i]))



        return -loss

        
    """def pdf(self, x, **params):
        if self.model_txpe == 'exponential_renato':
            lamb = params.get('lamb')
            return lamb*torch.exp(-lamb*x)
        else:
            raise ValueError('Distribution not found')"""

class NeuralNetwork(nn.Module):
    def __init__(self,model_input,model_output,model_type,input_3d):
        super().__init__()
        self.model_type = model_type
        self.input_3d = input_3d    

        if input_3d:
            depth = 4  # Fixed depth as novos + latitutde + longitude + distance
        
            self.conv1 = nn.Conv1d(in_channels=depth, out_channels=1, kernel_size=depth, padding=1)
            self.flatten_size = model_input 
            self.layer1 = nn.Linear(self.flatten_size, 20) 
        else:
            self.layer1 = nn.Linear(model_input, 20)

        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 5)    
        self.layer4 = nn.Linear(5, model_output)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.input_3d:
            x = x.permute(0,2,1)
            x = torch.relu(self.conv1(x))
            x = x.view(x.size(0), -1) 

        out1 = self.layer1(x)
        out2 = self.relu(self.layer2(out1))
        out3 = self.relu(self.layer3(out2))
        out4 = self.layer4(out3)
        return out4
    
class NeuralNetworkExponential(nn.Module):
    def __init__(self,model_input,model_output,model_type,input_3d):
        super().__init__()
        self.model_type = model_type
        self.input_3d = input_3d    

        if input_3d:
            depth = 4  # Fixed depth as novos + latitude + longitude  + distance
            self.conv1 = nn.Conv1d(in_channels=depth, out_channels=1, kernel_size=depth, padding=1)
            self.flatten_size = model_input 
            self.layer1 = nn.Linear(self.flatten_size, 20) 
        else:
            self.layer1 = nn.Linear(model_input, 20)

        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 5)    
        self.layer4 = nn.Linear(5, model_output)
        self.exp = nn.Linear(5 + model_output, 1) # exponential distribution
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()   

    def forward(self, x):
        if self.input_3d:
            x = x.permute(0,2,1)
            x = torch.relu(self.conv1(x))
            x = x.view(x.size(0), -1) 

        out1 = self.layer1(x)
        out2 = self.relu(self.layer2(out1))
        out3 = self.relu(self.layer3(out2))
        logit = self.relu(self.layer4(out3))
        lamb = self.softplus(self.exp(torch.cat((out3, logit), dim=1)))  # concat weights and logits. Relu to avoid negative values
        return logit, lamb

class NeuralNetworkPareto(nn.Module):
    def __init__(self,model_input,model_output,model_type,input_3d):
        super().__init__()
        self.model_type = model_type
        self.input_3d = input_3d    

        if input_3d:
            depth = 4  # Fixed depth as novos + latitude + longitude  + distance
            self.conv1 = nn.Conv1d(in_channels=depth, out_channels=1, kernel_size=depth, padding=1)
            self.flatten_size = model_input 
            self.layer1 = nn.Linear(self.flatten_size, 20) 
        else:
            self.layer1 = nn.Linear(model_input, 20)

        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 5)    
        self.layer4 = nn.Linear(5, model_output)
        self.pareto_layer = nn.Linear(5 + model_output, 1) #  
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()   


    def forward(self, x):
        if self.input_3d:
            x = x.permute(0,2,1)
            x = torch.relu(self.conv1(x))
            x = x.view(x.size(0), -1) 

        out1 = self.layer1(x)
        out2 = self.relu(self.layer2(out1))
        out3 = self.relu(self.layer3(out2))
        logit = self.layer4(out3)
        alpha = self.pareto_layer(torch.cat((out3, logit), dim=1)) # concat weights and logits. To use class instead, change to torch.cat((out3, logit.argmax(1)), dim=1) and self.exp
        return logit, alpha

class linear_pytorch(nn.Module):
    def __init__(self, model_input):
        super(linear_pytorch, self).__init__()
        self.layer1 = nn.Linear(model_input , 1)
        self.model_type = 'linear_pytorch'
        
    def forward(self, x):
        return self.layer1(x)
    

class LogisticRegression(nn.Module):
    def __init__(self, model_input,input_3d, model_type):
        super(LogisticRegression, self).__init__()
        self.model_type = model_type
        self.input_3d = input_3d
        if self.input_3d:
            depth = 4  # Fixed depth as novos + latitude +longitude  + distance
            self.conv1 = nn.Conv1d(in_channels=depth, out_channels=1, kernel_size=depth, padding=1)
            self.flatten_size = model_input 
            self.layer1 = nn.Linear(self.flatten_size, 20) 
        else:
            self.layer1 = nn.Linear(model_input, 1)  

    def forward(self, x):
        if self.input_3d:
            x = x.permute(0,2,1)
            x = torch.relu(self.conv1(x))
            x = x.view(x.size(0), -1)
        if self.model_type == 'logistic':
            return torch.sigmoid(self.layer1(x)) 
        elif self.model_type == 'linear_regressor':
            return self.layer1(x)


class TransformerSelector(BaseEstimator, TransformerMixin):
            """Custom transformer to select between RBF, Polynomial, or None."""
            def __init__(self, transformer=None):
                self.transformer = transformer

            def fit(self, X, y=None):
                if self.transformer is not None:
                    self.transformer.fit(X, y)
                return self

            def transform(self, X):
                if self.transformer is not None:
                    return self.transformer.transform(X)
                return X

            def fit_transform(self, X, y=None):
                self.fit(X, y)
                return self.transform(X)
