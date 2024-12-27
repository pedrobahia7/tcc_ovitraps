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
    experiment_name = 'random_forest_3c' # Name of the experiment to be saved in mlflow
    iterations_ignore = 0

    # Model parameters
    # 'classifier' or 'regressor' or 'exponential_renato' or 'linear_regressor' or 'GAM' or 
    # 'Naive' or 'logistic'  or 'mlp' or 'random_forest' or 'svm' or 'catboost' or 
    # 'logistic_3c' or 'Naive_3c' or  'random_forest_3c' or 'svm_3c' or 'catboost_3c'
    
    models = ['random_forest_3c'] 
    
    # Input parameters
    lags = 5
    neigh_num = 10
    use_trap_info = True
    scale = False
    input_3d = False
    bool_input = False
    input_3_class = False
    truncate_100 = False
    cylindrical_input = True
    add_constant = True
    select_features = False # If True, feature selection will be performed 
    type_of_selection = 'backward' # 'forward', 'backward' or 'stepwise
    all_cols = False # If True, all columns will be used, if False, automatic feature selection will be used
    # Features to be used in the model
    info_cols = [] 
    """[ 'trap0_lag1', 'trap0_lag2', 'trap0_lag3', 'trap0_lag4', 'trap0_lag5', 'latitude0', 'longitude0', 'days0_lag1', 
                 'days0_lag2', 'days0_lag3', 'days0_lag4', 'days0_lag5', 'trap1_lag1', 'trap1_lag2', 'trap1_lag3', 'trap1_lag4', 'trap1_lag5', 
                 'days1_lag1', 'days1_lag2', 'days1_lag3', 'days1_lag4', 'days1_lag5', 'trap2_lag1', 'trap2_lag2', 'trap2_lag3', 
                 'trap2_lag4', 'trap2_lag5',  'days2_lag1', 'days2_lag2', 'days2_lag3', 'days2_lag4', 'days2_lag5', 'trap3_lag1', 
                 'trap3_lag2', 'trap3_lag3', 'trap3_lag4', 'trap3_lag5',  'days3_lag1', 'days3_lag2', 'days3_lag3', 'days3_lag4', 
                 'days3_lag5', 'trap4_lag1', 'trap4_lag2', 'trap4_lag3', 'trap4_lag4', 'trap4_lag5',  'days4_lag1', 'days4_lag2', 
                 'days4_lag3', 'days4_lag4', 'days4_lag5', 'trap5_lag1', 'trap5_lag2', 'trap5_lag3', 'trap5_lag4', 'trap5_lag5',  
                 'days5_lag1', 'days5_lag2', 'days5_lag3', 'days5_lag4', 'days5_lag5', 'trap6_lag1', 'trap6_lag2', 'trap6_lag3', 'trap6_lag4', 'trap6_lag5',
                    'days6_lag1', 'days6_lag2', 'days6_lag3', 'days6_lag4', 'days6_lag5', 'trap7_lag1', 'trap7_lag2', 'trap7_lag3', 
                   'trap7_lag4', 'trap7_lag5',  'days7_lag1', 'days7_lag2', 'days7_lag3', 'days7_lag4', 'days7_lag5', 'trap8_lag1', 
                   'trap8_lag2', 'trap8_lag3', 'trap8_lag4', 'trap8_lag5',  'days8_lag1', 'days8_lag2', 'days8_lag3', 'days8_lag4', 
                   'days8_lag5', 'trap9_lag1', 'trap9_lag2', 'trap9_lag3', 'trap9_lag4', 'trap9_lag5',  'days9_lag1', 'days9_lag2', 
                   'days9_lag3', 'days9_lag4', 'days9_lag5', 'zero_perc', 'semepi', 'temp_expo', 'semepi2', 'sin_semepi', 'sin_mesepi', 
                   'Temperatura_previsao', 'Precipitacao_previsao', 'Umidade_previsao', 'Temperatura_week_bfr_mean', 'Precipitacao_week_bfr_mean', 
                   'Umidade_week_bfr_mean']"""

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

    # Random Forest parameters
    n_estimators = 100
    max_depth = 10
    min_samples_split = 10
    min_samples_leaf = 2
    bootstrap= True
    grid_search_rf = True # If True, grid search will be performed to find the best hyperparameters


    # SVM parameters
    kernel = 'rbf'
    gamma = 'scale'
    alpha = 0.0001  
    grid_search_svm = True # If True, grid search will be performed to find the best hyperparameters

    # Catboost parameters
    catboost_iterations = 1000
    catboost_learning_rate = 0.1
    catboost_depth = 6
    catboost_l2_leaf_reg = 3
    catboost_random_strength = 1
    catboost_grid_search = True # If True, grid search will be performed to find the best hyperparameters
    

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
        'input_3_class': input_3_class,
        'info_cols': info_cols  


        

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

    parameters['rf_params'] = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'grid_search': grid_search_rf,
        'min_samples_leaf':min_samples_leaf,
        'bootstrap':bootstrap,
    }

    parameters['svm_params'] = {
        'kernel': kernel,
        'gamma': gamma,
        'alpha': alpha,
        'grid_search': grid_search_svm
    }

    parameters['catboost_params'] = {
        'iterations' : catboost_iterations,
        'learning_rate' : catboost_learning_rate,
        'depth' : catboost_depth,
        'l2_leaf_reg' : catboost_l2_leaf_reg,
        'random_strength' : catboost_random_strength,
        'grid_search' : catboost_grid_search
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
