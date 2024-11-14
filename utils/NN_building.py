import sklearn.metrics
import torch
from torch.utils.data import Dataset
from torch import nn
import pandas as pd
from typing import Tuple
import utils.NN_building as NN_building
import utils.NN_arquitectures as NN_arquitectures
from torch.utils.data import DataLoader
import os
import utils.NN_preprocessing as NN_preprocessing
import utils.generic as generic
import mlflow
import mlflow.pytorch
import numpy as np
import pdb
import sklearn
from sklearn.linear_model import LogisticRegression as scikit_LogisticRegression






def create_dataset(parameters:dict, data_path:str= None)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    ntraps = parameters['ntraps']
    lags = parameters['lags']
    model_type = parameters['model_type']
    use_trap_info = parameters['use_trap_info']
    random_split = parameters['random_split']
    test_size = parameters['test_size']
    scale = parameters['scale']
    input_3d = parameters['input_3d']
    bool_input = parameters['bool_input']

    if use_trap_info == False:
        assert input_3d == False , '3D input is only available if trap information is used'
    if data_path is None:
        data_path = f'./results/final_dfs/final_df_lag{lags}_ntraps{ntraps}.parquet'
    if os.path.exists(data_path):
        # data import and preprocessing
        data = pd.read_parquet(data_path)
        unnamed_cols = data.columns [['Unnamed' in col for col in data.columns] ] #TODO create function to load data
        data.drop(unnamed_cols,axis=1,inplace = True)
    else:
        data = NN_preprocessing.create_final_matrix(lags,ntraps) # TODO add perc zero and mesepid

    nplaca_index = data['nplaca']
    data.drop(columns=['nplaca'], inplace=True)
    ovos_flag = data['novos'].apply(lambda x: 1 if x > 0 else 0)#.rename('ovos_flag', inplace=True)
    
    #data['zero_perc'] = 1 - data['zero_perc'] TODO: create flag one_perc 

    # divide columns into groups
    days_columns = [f'days{i}_lag{j}' for i in range(ntraps) for j in range(1, lags+1)]
    eggs_columns = [f'trap{i}_lag{j}' for i in range(ntraps) for j in range(1, lags+1)]
    lat_columns = [f'latitude{i}' for i in range(ntraps)]
    long_columns = [f'longitude{i}' for i in range(ntraps)]

    info_cols = days_columns + lat_columns + long_columns + ['mesepid'] + ['zero_perc']
    
    #transform values to 0 and 1
    if bool_input:
        transformed_data = data[eggs_columns].map(lambda x: 1 if x > 0 else 0)
        data[eggs_columns] = transformed_data
    
    x, y = xy_definition(model_type, data, use_trap_info, ovos_flag, info_cols)

    # train test split
    x_train, x_test, y_train, y_test = NN_preprocessing.data_train_test_split(x, y, test_size, random_split,ovos_flag)
    # scaling
    if scale:
        x_train, x_test, y_train, y_test = NN_preprocessing.scale_dataset(x_train.copy(), 
                                            x_test.copy(), y_train.copy(), y_test.copy(), model_type, use_trap_info, eggs_columns, lat_columns,long_columns, days_columns)
    #convert to numpy with the correct shape
    if input_3d:
        x_train = NN_preprocessing.create_3d_input(x_train, ntraps, lags)
        x_test = NN_preprocessing.create_3d_input(x_test, ntraps, lags)
        y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
    else:
        x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    return x_train, x_test, y_train, y_test, nplaca_index

def xy_definition(model_type:str, data:pd.DataFrame, use_trap_info:bool, ovos_flag:pd.DataFrame,
                  info_cols:list)->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fucntion to define the x and y variables according to the model type.

    Parameters:
    model_type: string with the model type: classifier, regressor, exponential_renato, linear_regressor, logistic
    data: dataframe with the data
    use_trap_info: flag to use the traps information like days and distances
    ovos_flag: dataframe with booleans indicating the presence of eggs
    inf_col: list with the name of columns of days, latitude, longitude, mesepid and novos

    Returns:
    x: dataframe with the input variables
    y: dataframe with the output variables


    """
# definition of x and y
    if model_type == 'classifier':
        y = ovos_flag
    elif model_type == 'logistic':
        y = ovos_flag
    elif model_type == 'regressor' or model_type == 'linear_regressor':
        y = data['novos']
    elif model_type == 'exponential_renato' or model_type == 'pareto':
        y = pd.concat([ovos_flag.rename('ovos_flag', inplace=True),data['novos']],axis=1)

    data = data.drop(columns=['novos'])
    if use_trap_info == False:
        x = data.drop(columns=info_cols)
    else:
        x = data

    return x, y

def transform_data_to_tensor(x_train: np.array, x_test: np.array, y_train: np.array, y_test: np.array, model_type: str, device: str)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transform numpy arrays to tensors and send them to the device

    Parameters:
    x_train: numpy array with the training data
    x_test: numpy array with the test data
    y_train: numpy array with the training data
    y_test: numpy array with the test data
    model_type: classifier, regressor, exponential_renato, linear_regressor or logistic
    device: device to send the tensors

    Returns:
    xtrain: tensor with the training data
    xtest: tensor with the test data
    ytrain: tensor with the training data
    ytest: tensor with the test data    
    """

    if model_type == 'classifier' or model_type == 'logistic':
        output_type = torch.long
    elif model_type == 'regressor' or model_type == 'exponential_renato' or 'linear_regressor' or model_type == 'pareto':
        output_type = torch.float32

    xtrain = torch.tensor(x_train, dtype=torch.float32).to(device)
    xtest = torch.tensor(x_test, dtype=torch.float32).to(device)
    ytrain = torch.tensor(y_train, dtype=output_type).to(device)
    ytest = torch.tensor(y_test, dtype=output_type).to(device)

    return xtrain, xtest, ytrain, ytest

class CustomDataset(Dataset):
    def __init__(self, features, targets, model_type):
        self.features  = features.clone().detach().float()
        if model_type == 'classifier' or model_type == 'logistic':
            self.targets =  targets.clone().detach().long()
        elif model_type == 'regressor' or model_type == 'linear_regressor':
            self.targets = targets.clone().detach().float()
        elif model_type == 'exponential_renato' or model_type == 'pareto':
            self.targets = targets.clone().detach().float()



    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def input_output_sizes(lags, ntraps, use_trap_info,model_type,input_3d):
    # Network structure
    if use_trap_info:
        if input_3d:
            model_input = lags*ntraps # dimension of traps. depth (days and distances) is fixed 
        else:
            model_input = lags*ntraps + ntraps*2 + ntraps*lags # sum  of eggs, lats and long, and days
    else:
        model_input = lags*ntraps
        
    if model_type == 'classifier' or model_type == 'exponential_renato' or model_type == 'logistic' or model_type == 'pareto':
        model_output = 2
    elif model_type == 'regressor' or model_type == 'linear_regressor':
        model_output = 1
    return model_input, model_output

def define_model(model_type, model_input, model_output, input_3d,device):
    if  model_type == 'logistic':
        model = scikit_LogisticRegression(penalty=None, max_iter=5000)
    elif model_type == 'linear_regressor':
        model = NN_arquitectures.LogisticRegression(model_input,input_3d, model_type).to(device)
    elif model_type == 'exponential_renato':
        model = NN_arquitectures.NeuralNetworkExponential(model_input, model_output,model_type,input_3d).to(device)
    elif model_type == 'pareto':
        model = NN_arquitectures.NeuralNetworkPareto(model_input, model_output,model_type,input_3d).to(device)
    else:
        model = NN_arquitectures.NeuralNetwork(model_input, model_output,model_type,input_3d).to(device)
    return model

def define_loss_functions(model_type):
    if model_type == 'classifier':
        loss_func_class = nn.CrossEntropyLoss()
        loss_func_reg = None
    elif model_type == 'logistic':
        loss_func_class = None
        loss_func_reg = None
    elif model_type == 'regressor' or model_type == 'linear_regressor':
        loss_func_class = None
        loss_func_reg = nn.MSELoss()
    elif model_type == 'exponential_renato':
        loss_func_class = nn.CrossEntropyLoss()
        loss_func_reg = NN_arquitectures.ExponentialLoss()
    elif model_type =='pareto':
        loss_func_class = nn.CrossEntropyLoss()
        loss_func_reg = NN_arquitectures.ParetoLoss()

    return loss_func_class, loss_func_reg

def torch_accuracy(yref, yhat,model_type):
    if model_type == 'classifier' or model_type == 'logistic': #TODO check if this is correct
        return (yhat.argmax(1) == yref).type(torch.float).sum().item()
    if model_type == 'regressor' or model_type == 'linear_regressor':
        return ((torch.round(yhat) == yref).type(torch.float)).sum().item()

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

        loss_class, loss_reg, acc_class, acc_reg, error_reg = evaluate_NN(model.model_type,loss_func_class, loss_func_reg, yhat, ytrain) # depend on model type
        
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
            loss_class, loss_reg, acc_class, acc_reg, error_reg = evaluate_NN(model.model_type,loss_func_class, loss_func_reg, yhat, ytest) # depend on model type
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

def evaluate_NN(model_type,loss_func_class, loss_func_reg, yhat, y ):

    if model_type == 'logistic':

        loss_reg =  0
        loss_class =  0
        acc_class = sklearn.metrics.accuracy_score(y, yhat)
        acc_reg =  0 
        error_reg =  0
        total_loss = 0
        return total_loss, loss_class, loss_reg, acc_class, acc_reg, error_reg

    elif model_type == 'classifier':
        loss_class = loss_func_class(yhat.squeeze(), y) # yhat = logit
        loss_reg =  torch.tensor(0)
        acc_class = torch_accuracy(y,  yhat, model_type)
        acc_reg =  0
        error_reg =  0

    elif model_type == 'regressor' or model_type == 'linear_regressor':
        yhat = yhat.squeeze()
        loss_class =  torch.tensor(0)
        loss_reg = loss_func_reg(yhat, y)
        acc_class =  0
        acc_reg = torch_accuracy(y, yhat, model_type)
        error_reg = ((yhat - y)**2).sum().item()
    
    elif model_type == 'exponential_renato':
        logit, lamb = yhat
        # evaluate the pdf of the distribution
        
        yhat = torch.round(1/(lamb + 0.00001)) # rounded mean of the distribution
        # split NN output
        y_class =  y[:,0].long()
        y_reg = y[:,1]
        # calculate metrics
        loss_class = loss_func_class(logit, y_class)
        loss_reg = loss_func_reg(y_reg, lamb)
        acc_class = torch_accuracy(y_class, logit, 'classifier')
        acc_reg = torch_accuracy(y_reg, yhat, 'regressor')
        error_reg = ((yhat - y_reg)**2).sum().item() #mse

    elif model_type == 'pareto':

        logit, alpha = yhat
        x_m = 1 # TODO: if x_m must be implemented as a parameter, it must be passed in yhat 
        # evaluate the pdf of the distribution
        yhat = alpha*x_m/(alpha - 1) 

        y_class =  y[:,0].long()
        y_reg = y[:,1]
        # calculate metrics
        loss_class = loss_func_class(logit, y_class)
        loss_reg = loss_func_reg(y_reg, alpha)
        acc_class = torch_accuracy(y_class, logit, 'classifier')
        acc_reg = torch_accuracy(y_reg, yhat, 'regressor')
        error_reg = ((yhat - y_reg)**2).sum().item() #mse


    return loss_class, loss_reg, acc_class, acc_reg, error_reg


def create_history_dict():
    return {
        'total_loss': [],
        'loss_class': [],
        'loss_reg': [],
        'acc_class': [],
        'acc_reg': [],
        'error_reg':[]
    }

def append_history_dict(history_dict, results):
    total_loss, loss_class, loss_reg, acc_class, acc_reg, error_reg = results
    history_dict['total_loss'].append(total_loss)
    history_dict['loss_class'].append(loss_class)
    history_dict['loss_reg'].append(loss_reg)
    history_dict['acc_class'].append(acc_class)
    history_dict['acc_reg'].append(acc_reg)
    history_dict['error_reg'].append(error_reg)

def calc_model_output(model, xtest,loss_func_reg=None):

    if model.model_type == 'classifier' or model.model_type == 'logistic':
        yhat = model(xtest).argmax(1).cpu().numpy()
        return yhat
    elif model.model_type == 'regressor' or model.model_type == 'linear_regressor':
        yhat = model(xtest).round().cpu().detach().numpy() 
        return yhat.squeeze()
    elif model.model_type == 'exponential_renato':
        logit, lamb = model(xtest)
        yhat_reg = 1/lamb
        yhat_class = logit.argmax(1)
        yhat = torch.stack((yhat_class.unsqueeze(1), yhat_reg) ,dim=1)
        return yhat
    elif model.model_type == 'pareto':
        logit, alpha = model(xtest)
        x_m = 1 # TODO x_m must be passed as a parameter
        
        yhat_class = logit.argmax(1)
        yhat_reg = yhat_class
        for i in range(len(yhat_reg)):
            if alpha[i]> 1:
                yhat_reg[i] = alpha[i]*x_m/(alpha[i] - 1)*yhat_reg[i]
            else:
                yhat_reg[i] = 1000000*yhat_reg[i]
        yhat = torch.stack((yhat_class, yhat_reg) ,dim=1)
        return yhat
    else:
        raise ValueError('Model type not found')
    
def save_model_mlflow(parameters:dict, model, yhat,ytest, test_history, train_history, experiment_name = 'NN_ovitraps'):
    """
    Saves the model in the MLflow server.
    """

    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(experiment_name)


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
        mlflow.log_param('version', version)                                          # Log version


        #signature = mlflow.models.ModelSignature(inputs=x_train, outputs=y_train)
        if parameters['model_type'] == 'logistic':
            mlflow.sklearn.log_model(model, "model")#,signature=signature)                      # Log model
        
        else:
            mlflow.pytorch.log_model(model, "model")#,signature=signature)                      # Log model
            
