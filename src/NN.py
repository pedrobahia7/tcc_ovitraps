import sys
import os
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '.'))
sys.path.append(project_root)

import utils.generic as generic
import pdb
from tqdm import tqdm
import itertools
import src.NN_pipeline as NN_pipeline


if __name__ == '__main__':
    '''
    Define different parameters and call pipeline
    '''
# Parameters

    # Pipeline parameters
    repeat = 1  # Number of times the model will be trained and tested
    play_song = True
    stop_time = 2
    experiment_name = 'Feature_selection' # Name of the experiment to be saved in mlflow
    iterations_ignore = 0

    # Model parameters
    models = ['logistic']  # 'classifier' or 'regressor' or 'exponential_renato' or 'linear_regressor' or 'logistic' or 'GAM' or 'Naive' or 'mlp'
    
    # Input parameters
    lags = 14
    neigh_num = 20
    use_trap_info = True
    scale = True
    input_3d = False
    bool_input = True
    truncate_100 = False 
    cylindrical_input = False
    add_constant = True
    month_experiment = False # If True, ytrain will be divided by month and some random months will be selected for testing
    select_features = True


    # Train and Test split parameters
    split_type = 'year'
    test_size = 0.2 
    all_years_list = ['2011_12', '2012_13', '2013_14', '2014_15', '2015_16', '2016_17', '2017_18', '2018_19', '2019_20', '2020_21', '2021_22', '2022_23', '2023_24', '2024_25']
    year_list_train = ['2011_12', '2012_13', '2013_14', '2014_15', '2015_16', '2016_17', '2017_18', '2018_19', '2019_20', '2020_21', '2021_22'] # ['2011_12', ..., '2024_25'] be careful about the first and last year
    year_list_test = ['2022_23', '2023_24', '2024_25']


    

    # MLP parameters
    hidden_layers =  [(50,25,10,5)] 
    learning_rate_mlp = 'adaptive'
    activation =  'relu'  # Activation function
    learning_rate = 1e-2 #[2**(-10+i)*1e-3 for i in range(10)]
    batch_size = 100
    epochs = 10000
    tolerance = 1e-5
    n_iter_no_change = 30

    parameters = {
        'model_type': models,
        'use_trap_info': use_trap_info,
        'ntraps': neigh_num,
        'lags': lags,
        'split_type': split_type,
        'test_size': test_size,
        'scale': scale,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'max_epochs': epochs,
        'input_3d': input_3d,
        'bool_input': bool_input,
        'truncate_100': truncate_100,
        'cylindrical_input': cylindrical_input,
        'add_constant': add_constant,
        'year_list_train': year_list_train,
        'year_list_test': year_list_test,
        'month_experiment': month_experiment,
        'select_features': select_features,

        }
    
    parameters['mlp_params'] = {
        'hidden_layer_sizes': hidden_layers,  # Example: (50, 25, 25, 5) or (10,10,5)
        'max_iter': parameters['max_epochs'],  # Number of epochs
        'activation': activation,  # Activation function
        'solver': 'adam',  # Optimization solver
        'learning_rate': learning_rate_mlp,  # Learning rate schedule
        'n_iter_no_change': n_iter_no_change ,  # Early stopping criteria
        'shuffle': True,  # Shuffle training data
        'verbose': True,  # Print progress
        'early_stopping': False,  # Disable early stopping
        'tol': tolerance,  # Tolerance for optimization TODO
        #'learning_rate_init': parameters['learning_rate']  # Initial learning rate
    }

    total_iterations = len(models) * len(hidden_layers) * len(all_years_list)**2 * repeat - iterations_ignore
    with tqdm(total=total_iterations, desc="Combined Loops") as pbar:
        j = 0
        for i in range(repeat):
            for model in models:
                for layer in hidden_layers:
                    while j < iterations_ignore-1:
                        j += 1
                        continue
                    parameters['mlp_params']['hidden_layer_sizes'] = layer
                    parameters['model_type'] = model
                    print(f"Iteration {i} - Model {model} - Lags {parameters['lags']} - Neigh {parameters['ntraps']}")
                    NN_pipeline.pipeline(parameters, experiment_name=experiment_name)
                    pbar.update(1)


        if play_song: 
            generic.play_ending_song('./data/Sinfonia To Cantata # 29.mp3')
            generic.stop_ending_song(stop_time)
