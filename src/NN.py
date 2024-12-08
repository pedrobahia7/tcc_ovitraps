import sys
import os
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '.'))
sys.path.append(project_root)

import utils.generic as generic
import pdb
import tqdm
import itertools
import src.NN_pipeline as NN_pipeline


if __name__ == '__main__':
    '''
    Define different parameters and call pipeline
    '''
# Parameters

    # Pipeline parameters
    repeat = 1 # Number of times the model will be trained and tested
    play_song = False
    stop_time = 2
    experiment_name = 'Teste' # Name of the experiment to be saved in mlflow
    iterations_ignore = 0

    # Model parameters
    models = ['mlp']  # 'classifier' or 'regressor' or 'exponential_renato' or 'linear_regressor' or 'logistic' or 'GAM' or 'Naive' or 'mlp'
    
    # Input parameters
    lags = [5]
    neigh_num = [11]
    use_trap_info = True
    scale = True
    input_3d = False
    bool_input = False
    truncate_100 = True
    cylindrical_input = True
    add_constant = True


    # Train and Test split parameters
    split_type = 'year'
    test_size = 0.2 
    year_list_train = ["2012_13"] # ['2011_12', ..., '2024_25']
    year_list_test = ["2013_14"]

    

    # MLP parameters
    hidden_layers =  (25,10,10, 5) 
    learning_rate_mlp = ['adaptive']
    activation =  ['relu']  # Activation function
    learning_rate = [1e-2] #[2**(-10+i)*1e-3 for i in range(10)]
    batch_size = 100
    epochs = 10000
    tolerance = 1e-4
    n_iter_no_change = 50


    parameters = {
        'model_type': [],
        'use_trap_info': use_trap_info,
        'ntraps': [],
        'lags': [],
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


        }
    
    parameters['mlp_params'] = {
        'hidden_layer_sizes': hidden_layers,  # Example: (50, 25, 25, 5) or (10,10,5)
        'max_iter': parameters['max_epochs'],  # Number of epochs
        'activation': 'reLU',  # Activation function
        'solver': 'adam',  # Optimization solver
        'learning_rate': 'adaptive',  # Learning rate schedule
        'n_iter_no_change':n_iter_no_change ,  # Early stopping criteria
        'shuffle': True,  # Shuffle training data
        'verbose': True,  # Print progress
        'early_stopping': False,  # Disable early stopping
        'tol': tolerance,  # Tolerance for optimization TODO
        #'learning_rate_init': parameters['learning_rate']  # Initial learning rate
    }

    j = 0
    for i in range(repeat):
        for act in activation:
            for lr in learning_rate:
                for lr_mlp in learning_rate_mlp:
                    for model in models:
                        for lag, ntraps in tqdm.tqdm(itertools.product(lags, neigh_num),total=len(lags)*len(neigh_num)):
                                while j < iterations_ignore-1:
                                    j += 1
                                    continue
                                parameters['mlp_params']['learning_rate'] = lr_mlp
                                parameters['learning_rate'] = lr
                                parameters['model_type'] = model
                                parameters['lags'] = lag
                                parameters['ntraps'] = ntraps
                                parameters['mlp_params']['activation'] = act

                                print(f'Iteration {i} - Model {model} - Lags {lag} - Neigh {ntraps}')
                                NN_pipeline.pipeline(parameters, experiment_name=experiment_name)

    if play_song: 
        generic.play_ending_song()
        generic.stop_ending_song(stop_time)