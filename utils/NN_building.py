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



class CustomDataset(Dataset):
    def __init__(self, features, targets, model_type):
        self.features  = features.clone().detach().float()
        if model_type == 'classifier':
            self.targets =  targets.clone().detach().long()
        elif model_type == 'regressor':
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
    def __init__(self,model_input,model_output,model_type):
        super().__init__()
        self.model_type = model_type
        self.layer1 = nn.Linear(model_input, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 5)    
        self.layer4 = nn.Linear(5, model_output)
        if model_type == 'exponential_renato':
            self.exp = nn.Linear(5 + model_output, 1) # exponential distribution
      
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        if self.model_type == 'exponential_renato':
            lamb = self.exp(torch.cat((out3, out4), dim=1)) # concat weights and logits. To use class instead, change to torch.cat((out3, logit.argmax(1)), dim=1) and self.exp
            lamb = nn.ReLU()(lamb) + 0.00001 # noise added
            return out4, lamb
        return out4

def torch_accuracy(yref, yhat,model_type):
    if model_type == 'classifier':
        return (yhat.argmax(1) == yref).type(torch.float).sum().item()
    if model_type == 'regressor':
        return ((torch.round(yhat) == yref).type(torch.float)).sum().item()

def evaluate_NN(model,loss_func_class, loss_func_reg, yhat, y ):

    if model.model_type == 'classifier':
        loss_class = loss_func_class(yhat, y) # yhat = logit
        loss_reg =  torch.tensor(0)
        acc_class = torch_accuracy(y,  yhat, model.model_type)
        acc_reg =  0
        error_reg =  0

    elif model.model_type == 'regressor':
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

def transform_data_to_tensor(x_train, x_test, y_train, y_test, model_type, device):
    xtrain = torch.tensor(x_train.values, dtype=torch.float32).to(device)
    xtest = torch.tensor(x_test.values, dtype=torch.float32).to(device)
    if model_type == 'classifier':
        ytrain = torch.tensor(y_train.values, dtype=torch.long).to(device)
        ytest = torch.tensor(y_test.values, dtype=torch.long).to(device)
    elif model_type == 'regressor' or model_type == 'exponential_renato':
        ytrain = torch.tensor(y_train.values, dtype=torch.float32).to(device)
        ytest = torch.tensor(y_test.values, dtype=torch.float32).to(device)
    return xtrain, xtest, ytrain, ytest

def input_output_sizes(lags, ntraps, use_trap_info,model_type):
    # Network structure
    if use_trap_info:
        model_input = lags*ntraps + ntraps-1 + ntraps*lags # sum  of eggs, distances minus one and days
    else:
        model_input = lags*ntraps
        
    if model_type == 'classifier' or model_type == 'exponential_renato':
        model_output = 2
    elif model_type == 'regressor':
        model_output = 1
    elif model_type == 'exponential_renato':
        model_output = 2 # returns logit of cross entropy
                    # and lambda of exponential distribution
    return model_input, model_output

def xy_definition(model_type, data, use_trap_info, ovos_flag,days_columns,distance_columns):
# definition of x and y
    if model_type == 'classifier':
        y = ovos_flag
    elif model_type == 'regressor':
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
    elif model_type == 'regressor':
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

    if data_path is None:
        data_path = f'./results/final_dfs/final_df_lag{lags}_ntraps{ntraps}.csv'
    if os.path.exists(data_path):
        # data import and preprocessing
        data = pd.read_csv(data_path)
    else:
        data = NN_preprocessing.create_final_matrix(ntraps, lags)

    nplaca_index = data['nplaca']
    data.drop(columns=['nplaca','distance0'], inplace=True) # drop distance0 because it is always zero
    ovos_flag = data['novos'].apply(lambda x: 1 if x > 0 else 0)#.rename('ovos_flag', inplace=True)

    # divide columns into groups
    days_columns = [f'days{i}_lag{j}' for i in range(ntraps) for j in range(1, lags+1)]
    distance_columns = [f'distance{i}' for i in range(1,ntraps)]
    eggs_columns = [f'trap{i}_lag{j}' for i in range(ntraps) for j in range(1, lags+1)]

    x, y = xy_definition(model_type, data, use_trap_info, ovos_flag,days_columns,distance_columns)
    
    # train test split
    x_train, x_test, y_train, y_test = NN_preprocessing.data_train_test_split(x, y, test_size, random_split,ovos_flag)
    # scaling
    if scale:
        x_train, x_test, y_train, y_test = NN_preprocessing.scale_dataset(x_train.copy(), 
                                            x_test.copy(), y_train.copy(), y_test.copy(), model_type, use_trap_info, eggs_columns, distance_columns, days_columns)


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

    if model.model_type == 'classifier':
        yhat = model(xtest).argmax(1).cpu().numpy()
        return yhat
    elif model.model_type == 'regressor':
        yhat = model(xtest).round().cpu().detach().numpy() 
        return yhat.squeeze()
    elif model.model_type == 'exponential_renato':
        logit, lamb = model(xtest)
        param_dict = {'lamb': lamb}
        x_dist = torch.linspace(0, 5000, 5001)
        x_dist = x_dist.repeat(lamb.shape[0], 1)
        y_dist = loss_func_reg.pdf(x_dist, **param_dict) # pdf of the distribution is saved in the NN
        yhat_reg = y_dist.argmax(1)
        yhat_class = logit.argmax(1).cpu().numpy()
        return yhat_class, yhat_reg
    
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
 


def NN_pipeline(parameters:dict, data_path:str= None)->None:
    """
    Creates a neural network according to the parameters passed in the dictionary. 
    The dictionary must contain:
        model_type: classifier, regressor, exponential_renato
        use_trap_info: flag to use the traps information like days and distances
        ntraps: number of traps to be considered
        lags: number of lags to be considered
        random_split: flag to use random test/train split or not 
        test_size: percentage of the test set
        scale: flag to scale the data or not using the MinMaxScaler
        learning_rate: learning rate of the optimizer
        batch_size: batch size of the DataLoader
        epochs: number of epochs to train the model
    
    
    """

    model_type = parameters['model_type']
    use_trap_info = parameters['use_trap_info']
    ntraps = parameters['ntraps']
    lags = parameters['lags']
    random_split = parameters['random_split']
    learning_rate = parameters['learning_rate']
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")


    # create dataset

    x_train, x_test, y_train, y_test, nplaca_index = create_dataset(parameters, data_path)


    # transform to tensor
    xtrain, xtest, ytrain, ytest = transform_data_to_tensor(x_train, x_test, y_train, y_test, model_type, device)

    train_dataset = NN_building.CustomDataset(xtrain, ytrain,model_type)
    test_dataset = NN_building.CustomDataset(xtest, ytest,model_type)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=random_split)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=random_split)


    # Network structure
    model_input, model_output = input_output_sizes(lags, ntraps, use_trap_info,model_type)
    model = NeuralNetwork(model_input, model_output,model_type).to(device)

    # Loss functions
    loss_func_class, loss_func_reg = define_loss_functions(model_type)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    train_history = create_history_dict()
    test_history = create_history_dict()


    # Network Loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        total_loss_train, loss_class_train, loss_reg_train, acc_class_train, acc_reg_train, error_reg_train = train_loop(train_dataloader,
                                                                              model, loss_func_class, loss_func_reg, optimizer)        
        
        total_loss_test, loss_class_test, loss_reg_test, acc_class_test, acc_reg_test, error_reg_test = test_loop(test_dataloader,
                                                                                        model, loss_func_class, loss_func_reg)

        # Append Metrics
        append_history_dict(train_history, total_loss=total_loss_train, loss_class=loss_class_train, loss_reg=loss_reg_train, 
                            acc_class=acc_class_train, acc_reg=acc_reg_train, error_reg=error_reg_train)
        append_history_dict(test_history, total_loss=total_loss_test, loss_class=loss_class_test, loss_reg=loss_reg_test, 
                            acc_class=acc_class_test, acc_reg=acc_reg_test, error_reg=error_reg_test)
        
        torch.save(model.state_dict(), f'./results/NN/save_parameters/model{model_type}_lags{lags}_ntraps{ntraps}_epoch{t}.pth')

        
    print("Done!")
    generic.play_ending_song()
    generic.stop_ending_song(2)


    yhat = calc_model_output(model, xtest,loss_func_reg)

    save_model_mlflow(parameters, model, yhat,ytest, test_history, train_history)








   

"""

    if model_type == 'classifier':
        print(accuracy_score(y_test, yhat))
        print(confusion_matrix(y_test, yhat, normalize='true', labels=[0,1]))

"""
