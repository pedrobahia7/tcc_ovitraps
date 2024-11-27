import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import json
import plotly.graph_objects as go
import numpy as np
import utils.NN_building as NN_building



def get_runs_by_parameters(parameters):
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

def plot_results_pytorch(variable_plot, list_plot,version_list,size,epochs,mt):
    """
    variable_plot = ['total_loss', 'loss_class', 'loss_reg', 'acc_class', 'acc_reg', 'error_reg']
    
    Epochs must be used!
    """
 

    plt.figure(figsize=(15, 6))  
    for i in range(size):   
        y = list(map((lambda x: x/100 if x > 1 else x),list_plot[i][variable_plot]))
        x = range(1, epochs+1)
        plt.plot(y, x, label='Version {}'.format(version_list[i]))
        plt.xlabel('Epoch')
        plt.ylabel(f'{variable_plot}')
        plt.legend()
        plt.title(f'Model {variable_plot} {"TODO"}: {mt}')

    plt.show()

def info_from_artifacts(runs_df:pd.DataFrame,artifact_path:str):
    """ 
    Retrieve the logged dictionaries from artifacts
    
    Parameters:
    runs_df: DataFrame of runs containing the run_id for each analysed model
    artifact_path: path to the artifact to be retrieved ['test_history.json', 'train_history.json', 'output.json']

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

def surface_plot(z,ztitle):
    """
    Plot a 3D surface plot using Plotly

    Parameters:
    z: Pivoted DataFrame containing the values to be plotted 
    ztitle : Title of the z axis
    
    """
    z.index = z.index.astype(int)
    z.columns = z.columns.astype(int)
    z = z.sort_index(ascending=True)
    z = z.sort_index(axis =1,ascending=True)
    z = z.interpolate(method='linear', axis=0)
    fig = go.Figure(data=[go.Surface(z=z.values, x=z.columns, y=z.index)])

    # Update layout for better readability
    fig.update_layout(
        title="3D Surface Plot",
        scene=dict(
            xaxis_title='Lags (X)', 
            yaxis_title='Number of neighbors (Y)',
            zaxis_title= ztitle,

        ),
        coloraxis_colorbar=dict(title="Scale"),
        width=1000,  # Increase width
        height=800,   # Increase height

    )

    # Show plot
    fig.show()

def load_model(runs_df:pd.DataFrame, model_lib:str):
    """ 
    Load the model from the run id

    Parameters:
    runs_df: DataFrame of runs containing the run_id for each analysed model
    model_lib: library used to create model ['sklearn', 'pytorch']

    Returns:
    model: loaded model

    """
    model_uri = mlflow.artifacts.download_artifacts(run_id=runs_df['run_id'][0], artifact_path='model')
    if model_lib == 'sklearn':
        return mlflow.statsmodels.load_model(model_uri)
    elif model_lib == 'pytorch':
        return mlflow.pytorch.load_model(model_uri)
    elif model_lib == 'statsmodels':
        return mlflow.statsmodels.load_model(model_uri)
    else:
        raise ValueError('Model library not supported')
    
def compare_predictions(yhat, ytest):
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
