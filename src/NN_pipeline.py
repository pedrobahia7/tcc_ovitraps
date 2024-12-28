import sys
import os
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '.'))
sys.path.append(project_root)

import utils.NN_building as NN_building
import utils.NN_preprocessing as NN_preprocessing
import utils.mlflow_utils as mlflow_utils 

import torch
from torch.utils.data import DataLoader
import pdb
from sklearn.neural_network import MLPClassifier
import mlflow
import pandas as pd
import numpy as np
from typing import Tuple
import statsmodels.api as sm


def pipeline(parameters:dict, experiment_name:str = 'Teste', data_path:str= None)->None:
    """
    Creates a neural network according to the parameters passed in the dictionary. 
    The dictionary must contain:
        model_type: classifier, regressor, exponential_renato, linear_regressor, logistic, mlp, random_forest, Naive, GAM, svm, catboost, Naive_3c, logistic_3c, random_forest_3c, svm_3c, catboost_3c
        use_trap_info: flag to use the traps information like days and distances
        ntraps: number of traps to be considered
        lags: number of lags to be considered
        split_type: flag to  define the type of test/train split. Options: random, sequential, year
        test_size: percentage of the test set. Only used if split_type is random or sequential
        year_list_train: list of years to be used in the train set. Only used if split_type is year
        year_list_test: list of years to be used in the test set. Only used if split_type is year

        scale: flag to scale the data or not using the MinMaxScaler
        learning_rate: learning rate of the optimizer
        batch_size: batch size of the DataLoader
        epochs: number of epochs to train the model
        input_3d: flag to use 3D input or not
        bool_input: flag to use boolean input or not
        truncate_100: flag to truncate the data to 100 or not
        cylindrical_input: flag to use cylindrical input or not. Cylindrical input considers 5 lags of the trap and 2 lags of 10 closets traps
        add_constant: flag to add a constant to the input
        mlp_params: dictionary with the parameters of the MLP model. Only used if model_type is mlp



    
    Parameters

    parameters: dict containing the parameters to create the model
    data_path: str, path to the data file
    """
    # Check if the parameters are valid
    parameters = check_parameters(parameters)


    # create dataset
    x_train, x_test, y_train, y_test, index_dict = create_dataset(parameters, data_path)
    # Network structure
    model_input, model_output = NN_building.input_output_sizes(x_train, parameters['model_type'])

    # Loss functions
    loss_func_class, loss_func_reg = NN_building.define_loss_functions(parameters['model_type'])
    

    train_history = NN_building.create_history_dict()
    test_history = NN_building.create_history_dict()


    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(experiment_name)
    # begin model training
    with mlflow.start_run():

        if parameters['model_type'] in ['logistic', 'GAM'] :                
                if 'select_features' in parameters.keys() and parameters['select_features'] == True:
                    model, features = NN_building.select_model_stepwise(x_train, y_train,parameters, type= parameters['type_of_selection'])
                else:
                    if parameters['model_type'] in ['logistic']:
                        features = list(x_train.columns)
                        model = sm.Logit(y_train,x_train[features]).fit()
                    elif parameters['model_type'] in ['GAM']:
                        model, features = NN_building.GAM_model(x_train, y_train, parameters)

                yhat_train = (model.predict(x_train[features]) >= 0.5).astype(int)
                yhat = (model.predict(x_test[features]) >= 0.5).astype(int)
                NN_building.easy_save(train_history, test_history, yhat_train, y_train, yhat, y_test, 
                    parameters['model_type'],loss_func_class, loss_func_reg)

        elif parameters['model_type'] in ['logistic_3c']:
            features = list(x_train.columns)
            model = sm.MNLogit(y_train, x_train[features]).fit()
            yhat_train = (model.predict(x_train[features])).idxmax(axis=1)
            yhat = (model.predict(x_test[features])).idxmax(axis=1)
            NN_building.easy_save(train_history, test_history, yhat_train, y_train, yhat, y_test, 
                parameters['model_type'],loss_func_class, loss_func_reg)

        elif parameters['model_type'] in ['random_forest', 'random_forest_3c']:

            model, parameters = NN_building.random_forest_model(x_train, y_train, parameters)            
            yhat_train = model.predict(x_train)
            yhat = model.predict(x_test)
            NN_building.easy_save(train_history, test_history, yhat_train, y_train, yhat, y_test,   
                        parameters['model_type'],loss_func_class, loss_func_reg)
            features = list(x_train.columns)
                            
        elif parameters['model_type'] in ['svm', 'svm_3c']:
            model, parameters = NN_building.svm_model(x_train, y_train, x_test, y_test, parameters)
            yhat_train = model.predict(x_train)
            yhat = model.predict(x_test)
            NN_building.easy_save(train_history, test_history, yhat_train, y_train, yhat, y_test, 
                    parameters['model_type'],loss_func_class, loss_func_reg)
            features = list(x_train.columns)

        elif parameters['model_type'] in ['catboost', 'catboost_3c']:
            if 'mesepid' in x_train.columns:
                x_test['mesepid'] = x_test['mesepid'].astype(str)
                x_train['mesepid'] = x_train['mesepid'].astype(str)
            model, parameters = NN_building.catboost_model(x_train, y_train, parameters)
            yhat_train = model.predict(x_train)
            yhat = model.predict(x_test)
            NN_building.easy_save(train_history, test_history, yhat_train, y_train, yhat, y_test,
                    parameters['model_type'],loss_func_class, loss_func_reg)
            features = list(x_train.columns)

        elif parameters['model_type']in ['Naive']: 
            yhat_train = x_train['trap0_lag1'] 
            yhat = x_test['trap0_lag1'] 
            NN_building.easy_save(train_history, test_history, yhat_train, y_train, yhat, y_test, 
                    parameters['model_type'],loss_func_class, loss_func_reg)
            model = None
            features = None
        
        elif parameters['model_type'] in ['Naive_3c']: #TODO improvement needed. Number of days in exposure is fixed
            yhat_train = x_train['trap0_lag1']
            yhat = x_test['trap0_lag1']
            
            NN_building.easy_save(train_history, test_history, yhat_train, y_train, yhat, y_test, 
                    parameters['model_type'],loss_func_class, loss_func_reg)
            model = None
            features = None

        elif parameters['model_type'] in ['mlp']:
            model = MLPClassifier(**parameters['mlp_params'])

            model.fit(x_train, y_train)
            yhat = model.predict(x_test)
            yhat_train = model.predict(x_train)
            NN_building.easy_save(train_history, test_history, yhat_train, y_train, yhat, y_test, 
                    parameters['model_type'],loss_func_class, loss_func_reg)
            if model.solver !='lbfgs':
                train_history['loss_class'] = model.loss_curve_
            features = None
            
            #save parameters specific of MLP
            parameters['epochs'] =  model.n_iter_
            parameters['activation_function'] = parameters['mlp_params']['activation']
            parameters['hidden_layer_sizes'] = parameters['mlp_params']['hidden_layer_sizes']
            parameters['solver'] = parameters['mlp_params']['solver']
            parameters['learning_rate_type'] = parameters['mlp_params']['learning_rate']

        elif parameters['model_type'] in ['exponential_renato','classifier','regressor','pareto','linear_regressor'] : #Pytorch models

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            print(f"Using {device} device")

            # transform to tensor
            xtrain, xtest, ytrain, ytest = NN_building.transform_data_to_tensor(x_train, x_test, y_train, y_test, parameters['model_type'], device)

            train_dataset = NN_building.CustomDataset(xtrain, ytrain,parameters['model_type'])
            test_dataset = NN_building.CustomDataset(xtest, ytest,parameters['model_type'])
            train_dataloader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=parameters['train_test_split'])
            test_dataloader = DataLoader(test_dataset, batch_size=parameters['batch_size'], shuffle=parameters['train_test_split'])

            model = NN_building.define_model(parameters['model_type'], model_input, model_output, parameters['input_3d'],device)

            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'])
            # Network Loop
            for t in range(parameters['max_epochs']):
                print(f"Epoch {t+1}\n-------------------------------")
                NN_building.easy_save(train_history, test_history, yhat_train, y_train, yhat, y_test, 
                    parameters['model_type'],loss_func_class, loss_func_reg)
                torch.save(model.state_dict(), f"./results/NN/save_parameters/model{parameters['model_type']}_lags{parameters['lags']}_ntraps{parameters['ntraps']}_epoch{t}.pth")
                print(NN_building.calc_model_output(model, xtest,loss_func_reg).sum()/xtest.shape[0])
            
            print("Done!")
            
            yhat = NN_building.calc_model_output(model, xtest,loss_func_reg)
        
        else:
            raise ValueError('Model type not recognized')
        
        mlflow_utils.save_model_mlflow(parameters=parameters, model=model, ytrain=y_train,
                        yhat = yhat, ytest = y_test, test_history = test_history, 
                        train_history = train_history, features = features, index_dict = index_dict)
        
def check_parameters(parameters:dict)->dict:
    """
    Function to check if the parameters are valid.

    Parameters:
    parameters: dictionary with the parameters of the model

    Returns:
    parameters: dictionary with modified  parameters of the model
    """


    if parameters['use_trap_info'] == False:
        assert parameters['input_3d'] == False , '3D input is only available if trap information is used'
    
    if parameters['cylindrical_input']:
        assert (parameters['lags'] >= 5), 'Cylindrical input is only available for lags >= 5'
    assert not(parameters['truncate_100'] == True and parameters['bool_input'] == True), 'Truncate 100 and bool input cannot be true at the same time' 

    if parameters['model_type'] in ['Naive']:
        parameters['bool_input'] =True
        parameters['truncate_100'] = False
        #, 'Naive model currently only accepts bool input'
    
    if parameters['model_type'] in ['Naive_3c']:
        parameters['bool_input'] = False
        parameters['truncate_100'] = False
        parameters['scale'] = False
        parameters['input_3_class'] = True  

    if parameters['split_type'] == 'year':
        assert len(parameters['year_list_train']) > 0, 'Year list train must be defined for year split'
        assert len(parameters['year_list_test']) > 0, 'Year list test must be defined for year split'

    if 'info_cols' in parameters.keys():
         parameters['all_cols']==False, 'all_cols cant be used if info_cols is set manually'
    return parameters

def create_dataset(parameters:dict, data_path:str= None)->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Function to create the dataset for the neural network models. It reads the raw data given by PBH and preprocesses 
    it according to the model condfiguration and parameters.

    Parameters:
    parameters: dictionary with the parameters of the model
    data_path: string with the path to the data

    Returns:
    x_train: pd.DataFrame with the training input data
    x_test: pd.DataFrame with the test input data
    y_train: pd.DataFrame with the training target data
    y_test: pd.DataFrame with the test target data
    nplaca_index: dict with the index of the nplaca column for the train and test sets 
    """


    if data_path is None:
            data_path = f"./results/final_dfs/final_df_lag{parameters['lags']}_ntraps{parameters['ntraps']}.parquet"


    # create data
    if os.path.exists(data_path):
        # data import and preprocessing
        data = pd.read_parquet(data_path)
        unnamed_cols = data.columns [['Unnamed' in col for col in data.columns] ] #TODO create function to load data
        data.drop(unnamed_cols,axis=1,inplace = True)
    else:
        data = NN_preprocessing.create_final_matrix(parameters['lags'],parameters['ntraps'],save_path=data_path) # TODO add perc zero and mesepid

  
    if len(parameters['info_cols']) > 0 :
        info_cols = parameters['info_cols'] + ['anoepid','nplaca']
        eggs_columns = [f'trap{i}_lag{j}' for i in range(parameters['ntraps']) for j in range(1, parameters['lags']+1)]

    else:

        if parameters['all_cols'] == True:
            info_cols = [col for col in data.columns if col not in ['novos']]
            eggs_columns = [f'trap{i}_lag{j}' for i in range(parameters['ntraps']) for j in range(1, parameters['lags']+1)]

            
        else:
            #data['zero_perc'] = 1 - data['zero_perc'] TODO: create flag one_perc 
            features_to_add = [ 'zero_perc','mesepid']

        # divide columns into groups
            if parameters['cylindrical_input'] == False:
                #days_columns = [f'days{i}_lag{j}' for i in range(parameters['ntraps']) for j in range(1, parameters['lags']+1)]
                days_columns = []
                eggs_columns = [f'trap{i}_lag{j}' for i in range(parameters['ntraps']) for j in range(1, parameters['lags']+1)]
            elif parameters['cylindrical_input'] == True:
                #days_columns = [f'days{i}_lag{j}' for i in [0] for j in range(1, 6)]
                eggs_columns = [f'trap{i}_lag{j}' for i in [0] for j in range(1, min(11,parameters['lags']+1))]
                #days_columns += [f'days{i}_lag{j}' for i in range(1, parameters['ntraps']) for j in range(1, )]
                eggs_columns += [f'trap{i}_lag{j}' for i in range(1, parameters['ntraps']) for j in range(1, 3)]
                days_columns = []

            lat_column = ['latitude0']
            long_column = ['longitude0']

            info_cols  = eggs_columns + lat_column + long_column + features_to_add + ['nplaca', 'Temperatura_week_bfr_mean','Precipitacao_week_bfr_mean','Umidade_week_bfr_mean']
            # Temperatura_previsao	Precipitacao_previsao	Umidade_previsao	Temperatura_week_bfr_mean	Precipitacao_week_bfr_mean	Umidade_week_bfr_mean
            if parameters['split_type'] == 'year':
                info_cols += ['anoepid']

    #transform values to 0 and 1
    if parameters['bool_input']:
        transformed_data = data[eggs_columns].map(lambda x: 1 if x > 1 else x) 
        data[eggs_columns] = transformed_data
    
    if parameters['truncate_100']:
        transformed_data = data[eggs_columns].map(lambda x: 100 if x > 100 else x) 
        data[eggs_columns] = transformed_data

    if parameters['input_3_class']:
        transformed_data = data[eggs_columns].apply(lambda x: pd.cut(x/7, bins=[-float('inf'), 2.65, 5, float('inf')], labels=[0, 1, 2]))
        data[eggs_columns] = transformed_data



    
    # columns to be added as input
    x, y = NN_building.xy_definition( data=data, parameters = parameters,
                          info_cols=info_cols, eggs_cols=eggs_columns)
    # train test split
    x_train, x_test, y_train, y_test, index_dict = NN_preprocessing.data_train_test_split(x, y, parameters)
    
    # scaling
    if parameters['scale']:
        x_train, x_test, y_train, y_test, scaler_dict = NN_preprocessing.scale_dataset(x_train.copy(), 
                                x_test.copy(), y_train.copy(), y_test.copy(), parameters)
        
    #convert to numpy with the correct shape
    if parameters['input_3d']:
        x_train = NN_preprocessing.create_3d_input(x_train, parameters['ntraps'], parameters['lags'])
        x_test = NN_preprocessing.create_3d_input(x_test, parameters['ntraps'], parameters['lags'])
        y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

    if parameters['model_type'] not in ['logistic', 'random_forest', 'GAM', 'Naive', 'mlp', 'svm', 'catboost', 'logistic_3c', 'Naive_3c', 
                                        'random_forest_3c', 'svm_3c','catboost_3c']:    #return a numpy array instead of a df
        x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    return x_train, x_test, y_train, y_test, index_dict

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

        loss_class, loss_reg, acc_class, acc_reg, error_reg = NN_building.evaluate_NN(model.model_type,loss_func_class, loss_func_reg, yhat, ytrain) # depend on model type
        
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
            loss_class, loss_reg, acc_class, acc_reg, error_reg = NN_building.evaluate_NN(model.model_type,loss_func_class, loss_func_reg, yhat, ytest) # depend on model type
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
