import torch
from torch.utils.data import Dataset
from torch import nn
import pandas as pd
from typing import Tuple
import utils.NN_building as NN_building
from torch.utils.data import DataLoader
import os
import utils.NN_preprocessing as NN_preprocessing
import utils.generic as generic
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.pytorch
import numpy as np
import pdb




class CustomDataset(Dataset):
    def __init__(self, features, targets, model_type):
        self.features  = features.clone().detach().float()
        if model_type == 'classifier' or model_type == 'logistical':
            self.targets =  targets.clone().detach().long()
        elif model_type == 'regressor' or model_type == 'linear_regressor':
            self.targets = targets.clone().detach().float()
        elif model_type == 'exponential_renato':
            self.targets = targets.clone().detach().float()



    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class CustomLoss(nn.Module):
    """
    Class to create a custom loss and calculate the pdf of its original distribution if necessary. The name of the distribution must 
    be passed in the constructor
    
    Currently available distributions:
    exponential - **params['lamb']

    """
    def __init__(self, model_type:str):
        """
        Distribution: name of the distribution to be used as reference. 
        
        
        """
        super(CustomLoss, self).__init__()
        self.model_type = model_type

    def forward(self, y, **params):
        if self.model_type == 'exponential_renato':
            # Example: NNL of the exponential distribution
            lamb = params.get('lamb')
            loss = 0
            for i in range(len(lamb)):
                loss += torch.log(lamb[i]) - lamb[i]*y[i]
            return -loss
        else:
            raise ValueError('Distribution not found')
        
    def pdf(self, x, **params):
        if self.model_type == 'exponential_renato':
            lamb = params.get('lamb')
            return lamb*torch.exp(-lamb*x)
        else:
            raise ValueError('Distribution not found')

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

def torch_accuracy(yref, yhat,model_type):
    if model_type == 'classifier' or model_type == 'logistical':
        return (yhat.argmax(1) == yref).type(torch.float).sum().item()
    if model_type == 'regressor' or model_type == 'linear_regressor':
        return ((torch.round(yhat) == yref).type(torch.float)).sum().item()

def evaluate_NN(model,loss_func_class, loss_func_reg, yhat, y ):

    if model.model_type == 'classifier' or model.model_type == 'logistical':
        if model.model_type == 'logistical':
            loss_class = loss_func_class(yhat.squeeze(), y.type(torch.float))
        else:
            loss_class = loss_func_class(yhat.squeeze(), y) # yhat = logit
        loss_reg =  torch.tensor(0)
        acc_class = torch_accuracy(y,  yhat, model.model_type)
        acc_reg =  0
        error_reg =  0

    elif model.model_type == 'regressor' or model.model_type == 'linear_regressor':
        yhat = yhat.squeeze()
        loss_class =  torch.tensor(0)
        loss_reg = loss_func_reg(yhat, y)
        acc_class =  0
        acc_reg = torch_accuracy(y, yhat, model.model_type)
        error_reg = ((yhat - y)**2).sum().item()
    
    elif model.model_type == 'exponential_renato':
        logit, lamb = yhat
        param_dict = {'lamb': lamb}
        # evaluate the pdf of the distribution
        x_dist = torch.linspace(0, 5000, 5001)
        x_dist = x_dist.repeat(lamb.shape[0], 1)
        y_dist = loss_func_reg.pdf(x_dist, **param_dict) # pdf of the distribution is saved in the NN
        yhat = y_dist.argmax(1)
        # split NN output
        y_class =  y[:,0].long()
        y_reg = y[:,1]
        # calculate metrics
        loss_class = loss_func_class(logit, y_class)
        loss_reg = loss_func_reg(y_reg, **param_dict)
        acc_class = torch_accuracy(y_class, logit, 'classifier')
        acc_reg = torch_accuracy(y_reg, yhat, 'regressor')
        error_reg = ((yhat - y_reg)**2).sum().item() #mse

    return loss_class, loss_reg, acc_class, acc_reg, error_reg


# Create train and test loops
def train_loop(dataloader, model, loss_func_class, loss_func_reg, optimizer):
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss_train, loss_class_train, loss_reg_train, acc_class_train, acc_reg_train, error_reg_train  = 0, 0, 0, 0, 0, 0 

    for batch, (xtrain, ytrain) in enumerate(dataloader):
        # Compute prediction and loss
        optimizer.zero_grad()
        # Calculate loss

        yhat = model(xtrain)
        loss_class, loss_reg, acc_class, acc_reg, error_reg = evaluate_NN(model,loss_func_class, loss_func_reg, yhat, ytrain) # depend on model type
        
        # save losses
        total_loss = loss_class + loss_reg
        total_loss_train += total_loss.item()
        loss_class_train += loss_class.item()
        loss_reg_train += loss_reg.item()
        acc_class_train += acc_class
        acc_reg_train += acc_reg
        error_reg_train += error_reg

        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
    
    loss_class_train /= num_batches
    loss_reg_train /= num_batches
    total_loss_train = loss_class_train + loss_reg_train
    acc_class_train /= size
    acc_reg_train /= size
    error_reg_train /= size
        
    return total_loss_train, loss_class_train, loss_reg_train, acc_class_train, acc_reg_train, error_reg_train
    '''
        if batch % 100 == 0:
            loss, current = totalLoss.item(), batch * batch_size + len(xtest)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
'''

def test_loop(dataloader, model, loss_func_class, loss_func_reg):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss_test, loss_class_test, loss_reg_test, acc_class_test, acc_reg_test, error_reg_test  = 0, 0, 0, 0, 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for xtest, ytest in dataloader:
        # Calculate loss

            yhat = model(xtest)
            loss_class, loss_reg, acc_class, acc_reg, error_reg = evaluate_NN(model,loss_func_class, loss_func_reg, yhat, ytest) # depend on model type
            # save losses
            loss_class_test += loss_class.item()
            loss_reg_test += loss_reg.item()
            total_loss_test +=  loss_class.item() + loss_reg.item()
            acc_class_test += acc_class
            acc_reg_test += acc_reg
            error_reg_test += error_reg

    loss_class_test /= num_batches
    loss_reg_test /= num_batches
    total_loss_test = loss_class_test + loss_reg_test
    acc_class_test /= size
    acc_reg_test /= size
    error_reg_test /= size

    return total_loss_test, loss_class_test, loss_reg_test, acc_class_test, acc_reg_test, error_reg_test

def transform_data_to_tensor(x_train: np.array, x_test: np.array, y_train: np.array, y_test: np.array, model_type: str, device: str)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transform numpy arrays to tensors and send them to the device

    Parameters:
    x_train: numpy array with the training data
    x_test: numpy array with the test data
    y_train: numpy array with the training data
    y_test: numpy array with the test data
    model_type: classifier, regressor, exponential_renato, linear_regressor or logistical
    device: device to send the tensors

    Returns:
    xtrain: tensor with the training data
    xtest: tensor with the test data
    ytrain: tensor with the training data
    ytest: tensor with the test data    
    """

    if model_type == 'classifier' or model_type == 'logistical':
        output_type = torch.long
    elif model_type == 'regressor' or model_type == 'exponential_renato' or 'linear_regressor':
        output_type = torch.float32

    xtrain = torch.tensor(x_train, dtype=torch.float32).to(device)
    xtest = torch.tensor(x_test, dtype=torch.float32).to(device)
    ytrain = torch.tensor(y_train, dtype=output_type).to(device)
    ytest = torch.tensor(y_test, dtype=output_type).to(device)

    return xtrain, xtest, ytrain, ytest

def input_output_sizes(lags, ntraps, use_trap_info,model_type,input_3d):
    # Network structure
    if use_trap_info:
        if input_3d:
            model_input = lags*ntraps # dimension of traps. depth (days and distances) is fixed 
        else:
            model_input = lags*ntraps + ntraps-1 + ntraps*lags # sum  of eggs, distances minus one and days
    else:
        model_input = lags*ntraps
        
    if model_type == 'classifier' or model_type == 'exponential_renato' or model_type == 'logistical':
        model_output = 2
    elif model_type == 'regressor' or model_type == 'linear_regressor':
        model_output = 1
    return model_input, model_output

def xy_definition(model_type, data, use_trap_info, ovos_flag,days_columns,distance_columns):
# definition of x and y
    if model_type == 'classifier' or model_type == 'logistical':
        y = ovos_flag
    elif model_type == 'regressor' or model_type == 'linear_regressor':
        y = data['novos']
    elif model_type == 'exponential_renato':
        y = pd.concat([ovos_flag.rename('ovos_flag', inplace=True),data['novos']],axis=1)

    if use_trap_info:
        x = data.drop(columns=['novos'])
    else:
        drop_cols = ['novos'] + days_columns + distance_columns
        x = data.drop(columns=drop_cols)
    return x, y

def define_loss_functions(model_type):
    if model_type == 'classifier':
        loss_func_class = nn.CrossEntropyLoss()
        loss_func_reg = None
    elif model_type == 'logistical':
        loss_func_class = nn.BCEWithLogitsLoss()
        loss_func_reg = None
    elif model_type == 'regressor' or model_type == 'linear_regressor':
        loss_func_class = None
        loss_func_reg = nn.MSELoss()
    elif model_type == 'exponential_renato':
        loss_func_class = nn.CrossEntropyLoss()
        loss_func_reg = NN_building.CustomLoss(model_type)
    return loss_func_class, loss_func_reg

def create_dataset(parameters:dict, data_path:str= None)->Tuple[DataLoader, DataLoader]:
    ntraps = parameters['ntraps']
    lags = parameters['lags']
    model_type = parameters['model_type']
    use_trap_info = parameters['use_trap_info']
    random_split = parameters['random_split']
    test_size = parameters['test_size']
    scale = parameters['scale']
    input_3d = parameters['input_3d']

    if use_trap_info == False:
        assert input_3d == False , '3D input is only available if trap information is used'

    if data_path is None:
        data_path = f'./results/final_dfs/final_df_lag{lags}_ntraps{ntraps}.csv'
    if os.path.exists(data_path):
        # data import and preprocessing
        data = pd.read_csv(data_path)
    else:
        data = NN_preprocessing.create_final_matrix(ntraps, lags)

    nplaca_index = data['nplaca']
    data.drop(columns=['nplaca'], inplace=True)
    ovos_flag = data['novos'].apply(lambda x: 1 if x > 0 else 0)#.rename('ovos_flag', inplace=True)

    # divide columns into groups
    days_columns = [f'days{i}_lag{j}' for i in range(ntraps) for j in range(1, lags+1)]
    distance_columns = [f'distance{i}' for i in range(ntraps)]
    eggs_columns = [f'trap{i}_lag{j}' for i in range(ntraps) for j in range(1, lags+1)]

    x, y = xy_definition(model_type, data, use_trap_info, ovos_flag,days_columns,distance_columns)
    
    if use_trap_info == True and input_3d == False:
            x.drop(columns=['distance0'], inplace=True)

    # train test split
    x_train, x_test, y_train, y_test = NN_preprocessing.data_train_test_split(x, y, test_size, random_split,ovos_flag)
    # scaling
    if scale:
        x_train, x_test, y_train, y_test = NN_preprocessing.scale_dataset(x_train.copy(), 
                                            x_test.copy(), y_train.copy(), y_test.copy(), model_type, use_trap_info, eggs_columns, distance_columns, days_columns)
    #convert to numpy with the correct shape
    if input_3d:
        x_train = NN_preprocessing.create_3d_input(x_train, ntraps, lags)
        x_test = NN_preprocessing.create_3d_input(x_test, ntraps, lags)
        y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
    else:
        x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


    return x_train, x_test, y_train, y_test, nplaca_index

def create_history_dict():
    return {
        'total_loss': [],
        'loss_class': [],
        'loss_reg': [],
        'acc_class': [],
        'acc_reg': [],
        'error_reg':[]
    }

def append_history_dict(history_dict, *, total_loss, loss_class, loss_reg, acc_class, acc_reg, error_reg):
    history_dict['total_loss'].append(total_loss)
    history_dict['loss_class'].append(loss_class)
    history_dict['loss_reg'].append(loss_reg)
    history_dict['acc_class'].append(acc_class)
    history_dict['acc_reg'].append(acc_reg)
    history_dict['error_reg'].append(error_reg)

def calc_model_output(model, xtest,loss_func_reg=None):

    if model.model_type == 'classifier' or model.model_type == 'logistical':
        yhat = model(xtest).argmax(1).cpu().numpy()
        return yhat
    elif model.model_type == 'regressor' or model.model_type == 'linear_regressor':
        yhat = model(xtest).round().cpu().detach().numpy() 
        return yhat.squeeze()
    elif model.model_type == 'exponential_renato':
        logit, lamb = model(xtest)
        param_dict = {'lamb': lamb}
        x_dist = torch.linspace(0, 5000, 5001)
        x_dist = x_dist.repeat(lamb.shape[0], 1)
        y_dist = loss_func_reg.pdf(x_dist, **param_dict) # pdf of the distribution is saved in the NN
        yhat_reg = y_dist.argmax(1)
        yhat_class = logit.argmax(1)
        yhat = torch.stack((yhat_class, yhat_reg) ,dim=1)

        return yhat
    
def save_model_mlflow(parameters:dict, model, yhat,ytest, test_history, train_history ):
    """
    Saves the model in the MLflow server.
    """

    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('NN_ovitraps')


    # Construct the filter to check model versio
    filter_string = " and ".join([f"params.{key} = '{value}'" for key, value in parameters.items()])

    # Search for existing runs using the constructed filter string
    existing_runs = mlflow.search_runs(filter_string=filter_string)
    version = len(existing_runs) + 1
    

    output = {
    'yhat': yhat.tolist(),
    'ytest': ytest.cpu().numpy().tolist(),
        }

    # Start an MLflow run
    with mlflow.start_run():
        # metrics
        metrics_dict = {
                        'Test Classification Accuracy': test_history['acc_class'][-1],
                        'Train Classification Accuracy': train_history['acc_class'][-1],
                        'Test Regression Accuracy': test_history['acc_reg'][-1],
                        'Train Regression Accuracy': train_history['acc_reg'][-1],
                        'Test Regression Error': test_history['error_reg'][-1],
                        'Train Regression Error': train_history['error_reg'][-1],
                        'Percentage of Zeros in Test': (ytest == 0).sum().item()/len(ytest)
                        }

        mlflow.log_metrics(metrics_dict)

        # historic results
        mlflow.log_dict( test_history,'test_history.json')
        mlflow.log_dict( train_history,'train_history.json')
        
        # Log parameters
        for key, value in parameters.items():
            mlflow.log_param(key, value)
        
        # Log outputs

        mlflow.log_dict(output, "output.json")                             

        
        #tags
        mlflow.set_tag('mlflow.runName', f"{parameters['model_type']}_lags{parameters['lags']}_ntraps{parameters['ntraps']}_version{version}")	
        mlflow.set_tag('model_type', parameters['model_type'])                       # Log model type
        mlflow.set_tag('ntraps', parameters['ntraps'])                               # Log number of traps
        mlflow.set_tag('lags', parameters['lags'])                                   # Log number of lags
        mlflow.set_tag('version', version)                                # Log version


        #signature = mlflow.models.ModelSignature(inputs=x_train, outputs=y_train)
        mlflow.pytorch.log_model(model, "model")#,signature=signature)                      # Log model
 




