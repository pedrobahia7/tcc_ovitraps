import sys
import os
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '.'))
sys.path.append(project_root)

import utils.NN_building as NN_building
import utils.NN_arquitectures as NN_arquitectures
import torch
from torch.utils.data import DataLoader
import pdb
import tqdm
import itertools

def NN_pipeline(parameters:dict, data_path:str= None)->None:
    """
    Creates a neural network according to the parameters passed in the dictionary. 
    The dictionary must contain:
        model_type: classifier, regressor, exponential_renato, linear_regressor, logistic
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
    input_3d = parameters['input_3d']


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # create dataset
    x_train, x_test, y_train, y_test, nplaca_index = NN_building.create_dataset(parameters, data_path)
    # transform to tensor
    xtrain, xtest, ytrain, ytest = NN_building.transform_data_to_tensor(x_train, x_test, y_train, y_test, model_type, device)
    train_dataset = NN_building.CustomDataset(xtrain, ytrain,model_type)
    test_dataset = NN_building.CustomDataset(xtest, ytest,model_type)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=random_split)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=random_split)
    # Network structure
    model_input, model_output = NN_building.input_output_sizes(lags, ntraps, use_trap_info,model_type,input_3d)
    model = NN_building.define_model(model_type, model_input, model_output, input_3d,device)

    # Loss functions
    loss_func_class, loss_func_reg = NN_building.define_loss_functions(model_type)
    


    train_history = NN_building.create_history_dict()
    test_history = NN_building.create_history_dict()

    if model_type == 'logistic':
            model.fit(xtrain, ytrain)
            yhat_train = model.predict(xtrain)
            yhat = model.predict(xtest)

            results_train = NN_building.evaluate_NN(model_type,loss_func_class, loss_func_reg, yhat_train, ytrain) # depend on model type
            results_test = NN_building.evaluate_NN(model_type,loss_func_class, loss_func_reg, yhat, ytest) # depend on model type
            NN_building.append_history_dict(train_history, results_train)
            NN_building.append_history_dict(test_history, results_test)
    

    else:
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Network Loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            
            results_train = NN_building.train_loop(train_dataloader, model, loss_func_class, loss_func_reg, optimizer)        
            
            results_test = NN_building.test_loop(test_dataloader,model, loss_func_class, loss_func_reg)

            # Append Metrics
            NN_building.append_history_dict(train_history, results_train)
            NN_building.append_history_dict(test_history, results_test)
            
            torch.save(model.state_dict(), f'./results/NN/save_parameters/model{model_type}_lags{lags}_ntraps{ntraps}_epoch{t}.pth')

        
        print("Done!")
        
        #generic.play_ending_song()
        #generic.stop_ending_song(2)


        yhat = NN_building.calc_model_output(model, xtest,loss_func_reg)
        
        torch.save(model.state_dict(), f'./results/NN/save_parameters/model{model_type}_lags{lags}_ntraps{ntraps}_final.pth')
    NN_building.save_model_mlflow(parameters, model, yhat, ytest, test_history, train_history,experiment_name = 'Classification')



if __name__ == '__main__':
    '''
    Define different parameters and call pipeline
    '''
    # Parameters

    repeat = 1 # Number of times the model will be trained and tested

    models = ['logistic']  # 'classifier' or 'regressor' or 'exponential_renato' or 'linear_regressor' or 'logistic'
    neigh_num = [1,2,3,5,8,10,13,15,17,20]
    lags = [1,3,5,8,10,13]  # max 13
    test_size = 0.2
    learning_rate =1e-3
    batch_size = 64
    epochs = 10
    use_trap_info = True
    scale = False
    random_split = False
    input_3d = False
    bool_input = True

    parameters = {
        'model_type': [],
        'use_trap_info': use_trap_info,
        'ntraps': [],
        'lags': [],
        'random_split': random_split,
        'test_size': test_size,
        'scale': scale,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'input_3d': input_3d,
        'bool_input': bool_input

        }


    bool_list = [True, False]

    """    for boolean1 in bool_list:
        parameters['use_trap_info'] = boolean1
        for boolean2 in bool_list:
            parameters['random_split'] = boolean2
            for type in metatron:
                parameters['model_type'] = type
        """
    for i in range(repeat):
        for model in models:
            parameters['model_type'] = model
            for lag, ntraps in tqdm.tqdm(itertools.product(lags, neigh_num),total=len(lags)*len(neigh_num)):
                parameters['lags'] = lag
                parameters['ntraps'] = ntraps
                print(f'Iteration {i} - Model {model} - Lags {lag} - Neigh {ntraps}')
                NN_pipeline(parameters)