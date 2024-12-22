import mlflow
import pandas as pd
import json
import numpy as np
import utils.NN_building as NN_building
import pdb
import datetime
import pickle
import torch



def get_runs_by_parameters(parameters:dict)->pd.DataFrame:
    """
    Get df with runs and theirs informations filtered by parameters

    Parameters:
    parameters: dictionary containing the parameters used to filter the runs

    Returns:
    runs_df: DataFrame containing the runs and their informations
    """
    # Construct the filter to check model version
    filter_string = " and ".join([f"params.{key} = '{value}'" for key, value in parameters.items()])
    # Search for existing runs using the constructed filter string
    runs_df = mlflow.search_runs(filter_string=filter_string)
    return runs_df

def get_runs_by_tags(tags):
    # Construct the filter to check model version
    filter_string = " and ".join([f"tag.{key} = '{value}'" for key, value in tags.items()])
    # Search for existing runs using the constructed filter string
    runs_df = mlflow.search_runs(filter_string=filter_string)
    return runs_df


def info_from_artifacts(runs_df:pd.DataFrame,artifact_path:str):
    """ 
    Retrieve the logged dictionaries from artifacts
    
    Parameters:
    runs_df: DataFrame of runs containing the run_id for each analysed model
    artifact_path: path to the artifact to be retrieved ['test_history.json', 'train_history.json', 'output.json', 'index_dict_test.json', 'index_dict_train.json']

    Returns:
    test_history_list: list of dictionaries containing the test history of each model
    train_history_list: list of dictionaries containing the train history of each model
    output_list: list of dictionaries containing the output of each model
    version_list: list of integers containing the version of each model    
    
    """

    info_history_list = []


    for _,run in (runs_df.iterrows()):
        run_id = run['run_id']

        info_history_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        with open(info_history_path, 'r') as f:
            info_history = json.load(f)
        info_history_list.append(info_history)
    return info_history_list



def get_version_list(runs_list):    
    """
    Retrieve the version of each model

    Parameters:
    runs_list: list of runs containing the run_id for each analysed model

    Returns:
    version_list: list of integers containing the version of each model

    """
    version_list = []
    for _,run in (runs_list.iterrows()):
        version_list.append(int(run['tags.version']))

    return version_list

def load_model(runs_df:pd.Series, model_lib:str):
    """ 
    Load the model from the run id

    Parameters:
    runs_df:  run containing the run_id for each analysed model
    model_lib: library used to create model ['sklearn', 'pytorch']

    Returns:
    model: loaded model

    """
    model_uri = mlflow.artifacts.download_artifacts(run_id=runs_df['run_id'], artifact_path='model')
    if model_lib == 'sklearn':
        return mlflow.statsmodels.load_model(model_uri)
    elif model_lib == 'pytorch':
        return mlflow.pytorch.load_model(model_uri)
    elif model_lib == 'statsmodels':
        return mlflow.statsmodels.load_model(model_uri)
    else:
        raise ValueError('Model library not supported')
    
def compare_predictions(yhat, ytest):
    """
    Compare the predictions with the test labels and return a boolean array with the results

    Parameters:
    yhat: predicted labels
    ytest: test labels

    Returns:
    bool_index: boolean array with the results of the comparison
    """
    # Initialize an empty boolean array
    bool_index = np.empty(0, dtype=bool)
    # Loop through yhat and ytest to compare elements
    for i in range(len(yhat)):
        if yhat[i] == ytest[i]:
            bool_index = np.concatenate([bool_index, [True]])
        else:
            bool_index = np.concatenate([bool_index, [False]])

    return bool_index


def read_parquet(parameters,data_path=None):
    """
    Read the parquet file and create the dataset for the neural network

    Parameters:
    parameters: dictionary containing the parameters to be used in the dataset creation
    data_path: path to the parquet file containing the data

    Returns:
    x_train: training dataset
    x_test: test dataset
    y_train: training labels
    y_test: test labels
    nplaca_index: index of the nplaca column
    info_cols: list of columns to be used in the dataset

    """
    lags = parameters['lags']
    ntraps = parameters['ntraps']
    
    if data_path is None:
        data_path = f'../results/final_dfs/final_df_lag{lags}_ntraps{ntraps}.parquet'

    origianl_df = pd.read_parquet(data_path)
    unnamed_cols = origianl_df.columns [['Unnamed' in col for col in origianl_df.columns] ]
    origianl_df.drop(unnamed_cols,axis=1,inplace = True)

    x_train, x_test, y_train, y_test, nplaca_index = NN_building.create_dataset(parameters, data_path )

    return x_train, x_test, y_train, y_test, nplaca_index

def save_model_mlflow(parameters:dict, model, ytrain:pd.DataFrame ,yhat:pd.DataFrame,ytest:pd.DataFrame, test_history, train_history, features, index_dict):

    """
    Saves the model in the MLflow server.
    """
    #  model

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "_").replace(":", "_")

    if parameters['model_type'] == 'logistic' or  parameters['model_type'] == 'mlp' or parameters['model_type'] == 'random_forest' or parameters['model_type'] == 'svm':
        pickle.dump(model, open(f"./results/NN/save_parameters/model{parameters['model_type']}_lags{parameters['lags']}_ntraps{parameters['ntraps']}_{now}.pkl", 'wb'))
    elif parameters['model_type'] == 'Naive':
        pass
    elif  parameters['model_type'] == 'GAM':
        model.save(f"./results/NN/save_parameters/model{parameters['model_type']}_lags{parameters['lags']}_ntraps{parameters['ntraps']}_{now}.pgam")
    else:
        torch.save(model.state_dict(), f"./results/NN/save_parameters/model{parameters['model_type']}_lags{parameters['lags']}_ntraps{parameters['ntraps']}_{now}.pth")

    # Construct the filter to check model versio
    filter_string = " and ".join([f"params.{key} = '{value}'" for key, value in parameters.items() if key not in ['mlp_params','year_list_test','year_list_train','info_cols'] ] )
    
    # Convert year lists to strings and add to filter string
    year_list_test = ','.join(map(str, parameters['year_list_test']))
    year_list_train = ','.join(map(str, parameters['year_list_train']))
    filter_string += f" and params.year_list_train = '{year_list_train}' and params.year_list_test = '{year_list_test}'"

    
    # Search for existing runs using the constructed filter string
    existing_runs = mlflow.search_runs(filter_string=filter_string)
    version = len(existing_runs) + 1
    

    output = {
    'yhat': yhat.tolist(),
    'ytest': ytest.tolist(),
        }

    # Start an MLflow run
    # metrics
    metrics_dict = {
                    'Test Classification Accuracy': test_history['acc_class'][-1],
                    'Train Classification Accuracy': train_history['acc_class'][-1],
                    'Test Regression Accuracy': test_history['acc_reg'][-1],
                    'Train Regression Accuracy': train_history['acc_reg'][-1],
                    'Test Regression Error': test_history['error_reg'][-1],
                    'Train Regression Error': train_history['error_reg'][-1],
                    'Percentage of Zeros in Test': 1 -(ytest == 1).sum().item()/len(ytest),
                    'Test Size': len(ytest),
                    'Train Size': len(ytrain)
                    }

    mlflow.log_metrics(metrics_dict)

    # historic results
    mlflow.log_dict( test_history,'test_history.json')
    mlflow.log_dict( train_history,'train_history.json')
    mlflow.log_dict(index_dict['test'].to_list(), "index_dict_test.json")                        
    mlflow.log_dict(index_dict['train'].to_list(), "index_dict_train.json")                        
    
    # Log parameters
    for key, value in parameters.items() :
        if key not in ['mlp_params','year_list_test','year_list_train']:
            mlflow.log_param(key, value)
    
    mlflow.log_param('year_list_train', year_list_train)
    mlflow.log_param('year_list_test', year_list_test)

    # Log outputs

    mlflow.log_dict(output, "output.json")     

    # Log features
    mlflow.log_param('features', features)

    
    #tags
    mlflow.set_tag('mlflow.runName', f"{parameters['model_type']}_lags{parameters['lags']}_ntraps{parameters['ntraps']}_version{version}")	
    mlflow.log_param('version', version)                                          # Log version


    # Log model
    if parameters['model_type'] == 'logistic' :
        mlflow.statsmodels.log_model(model, "model")#,signature=signature)                      # Log model
    elif parameters['model_type'] == 'GAM'or parameters['model_type'] == 'Naive':
        pass
    elif parameters['model_type'] == 'mlp' or parameters['model_type'] == 'random_forest' or parameters['model_type'] == 'svm':
        mlflow.sklearn.log_model(model, "model")
    else:
        mlflow.pytorch.log_model(model, "model")#,signature=signature)                      # Log model
  