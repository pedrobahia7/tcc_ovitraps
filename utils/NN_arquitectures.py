import torch
import torch.nn as nn
import numpy as np
# separate NN and create function eval 
class NeuralNetwork(nn.Module):
    def __init__(self,model_input,model_output,model_type,input_3d):
        super().__init__()
        self.model_type = model_type
        self.input_3d = input_3d    

        if input_3d:
            depth = 3  # Fixed depth as novos + day + distance
        
            self.conv1 = nn.Conv1d(in_channels=depth, out_channels=1, kernel_size=depth, padding=1)
            self.flatten_size = model_input 
            self.layer1 = nn.Linear(self.flatten_size, 20) 
        else:
            self.layer1 = nn.Linear(model_input, 20)

        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 5)    
        self.layer4 = nn.Linear(5, model_output)
        if model_type == 'exponential_renato':
            self.exp = nn.Linear(5 + model_output, 1) # exponential distribution

    def forward(self, x):
        if self.input_3d:
            x = x.permute(0,2,1)
            x = torch.relu(self.conv1(x))
            x = x.view(x.size(0), -1) 

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        if self.model_type == 'exponential_renato':
            lamb = self.exp(torch.cat((out3, out4), dim=1)) # concat weights and logits. To use class instead, change to torch.cat((out3, logit.argmax(1)), dim=1) and self.exp
            lamb = nn.ReLU()(lamb) + 0.00001 # noise added
            return out4, lamb
        return out4

class LogisticRegression(nn.Module):
    def __init__(self, model_input,input_3d, model_type):
        super(LogisticRegression, self).__init__()
        self.model_type = model_type
        self.input_3d = input_3d
        if self.input_3d:
            depth = 3  # Fixed depth as novos + day + distance
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
