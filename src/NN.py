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
    repeat = 5 # Number of times the model will be trained and tested
    play_song = True
    stop_time = 2
    experiment_name = 'Random_Forest' # Name of the experiment to be saved in mlflow
    iterations_ignore = 0

    # Model parameters
    # 'classifier' or 'regressor' or 'exponential_renato' or 'linear_regressor' or 'logistic' 
    # or 'GAM' or 'Naive' or 'mlp' or 'random_forest' or 'svm'
    
    models = ['random_forest']  
    
    # Input parameters
    lags = 14
    neigh_num = 10
    use_trap_info = True
    scale = True
    input_3d = False
    bool_input = False
    truncate_100 = True 
    cylindrical_input = True
    add_constant = True
    select_features = False # If True, feature selection will be performed 
    type_of_selection = 'backward' # 'forward', 'backward' or 'stepwise
    all_cols = False # If True, all columns will be used, if False, automatic feature selection will be used
    # Features to be used in the model
    #info_cols = [ 'latitude0', 'longitude0', 'mesepid', 'semepi', 'semepi2', 'sin_mesepi', 'sin_semepi', 'trap0_lag1', 'trap0_lag10', 'trap0_lag11', 'trap0_lag13', 'trap0_lag14', 'trap0_lag2', 'trap0_lag3', 'trap0_lag4', 'trap0_lag5', 'trap0_lag6', 'trap0_lag9', 'trap10_lag1', 'trap10_lag14', 'trap11_lag1', 'trap12_lag1', 'trap12_lag13', 'trap12_lag2', 'trap13_lag1', 'trap14_lag1', 'trap14_lag2', 'trap14_lag3', 'trap15_lag1', 'trap16_lag1', 'trap16_lag14', 'trap18_lag1', 'trap19_lag1', 'trap19_lag2', 'trap1_lag1', 'trap1_lag2', 'trap2_lag1', 'trap2_lag2', 'trap2_lag3', 'trap3_lag1', 'trap3_lag2', 'trap4_lag1', 'trap4_lag13', 'trap4_lag2', 'trap5_lag1', 'trap6_lag1', 'trap6_lag2', 'trap7_lag1', 'trap8_lag1', 'trap9_lag1', 'zero_perc']

    # Train and Test split parameters
    split_type = 'year'
    test_size = 0.2 
    all_years_list = ['2011_12', '2012_13', '2013_14', '2014_15', '2015_16', '2016_17', '2017_18', '2018_19', '2019_20', '2020_21', '2021_22', '2022_23', '2023_24', '2024_25']
    year_list_train = ['2011_12', '2012_13', '2013_14', '2014_15', '2015_16', '2016_17', '2017_18', '2018_19', '2019_20', '2020_21', '2021_22'] # ['2011_12', ..., '2024_25'] be careful about the first and last year
    year_list_test = ['2022_23', '2023_24', '2024_25']
    month_experiment = False # If True, ytrain will be divided by month and some random months will be selected for testing

    

    # MLP parameters
    hidden_layers =  [(50,25,10,5)] 
    learning_rate_mlp = 'adaptive'
    activation =  'relu'  # Activation function
    learning_rate = 1e-2 #[2**(-10+i)*1e-3 for i in range(10)]
    batch_size = 100
    epochs = 10000
    tolerance = 1e-5
    n_iter_no_change = 30

# Run pipeline
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
        'type_of_selection': type_of_selection,
        'all_cols': all_cols,
        #'info_cols': info_cols  


        

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

    total_iterations = len(models)  * repeat - iterations_ignore
    #* len(hidden_layers) * len(all_years_list)**2
    with tqdm(total=total_iterations, desc="Combined Loops") as pbar:
        j = 0
        for i in range(repeat):
            for model in models:
               # for layer in hidden_layers:
                    while j < iterations_ignore-1:
                        j += 1
                        continue
              #      parameters['mlp_params']['hidden_layer_sizes'] = layer
                    parameters['model_type'] = model
                    print(f"Iteration {i} - Model {model} - Lags {parameters['lags']} - Neigh {parameters['ntraps']}")
                    NN_pipeline.pipeline(parameters, experiment_name=experiment_name)
                    pbar.update(1)


        if play_song: 
            generic.play_ending_song('./data/Sinfonia To Cantata # 29.mp3')
            generic.stop_ending_song(stop_time)
