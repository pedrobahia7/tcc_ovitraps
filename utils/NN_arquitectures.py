import torch
import torch.nn as nn
import numpy as np
# from utils.NN_building 
# separate NN and create function eval 




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
        if self.model_type == 'logistical':
            return torch.sigmoid(self.layer1(x)) 
        elif self.model_type == 'linear_regressor':
            return self.layer1(x)
