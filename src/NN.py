import utils.NN_building as NN_building
import torch
from torch.utils.data import DataLoader



def NN_pipeline(parameters:dict, data_path:str= None)->None:
    """
    Creates a neural network according to the parameters passed in the dictionary. 
    The dictionary must contain:
        model_type: classifier, regressor, exponential_renato, linear_regressor, logistical
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
    if model_type == 'logistical' or model_type == 'linear_regressor':
        model = NN_building.LogisticRegression(model_input,input_3d, model_type).to(device)

    else:
        model = NN_building.NeuralNetwork(model_input, model_output,model_type,input_3d).to(device)

    # Loss functions
    loss_func_class, loss_func_reg = NN_building.define_loss_functions(model_type)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    train_history = NN_building.create_history_dict()
    test_history = NN_building.create_history_dict()


    # Network Loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        total_loss_train, loss_class_train, loss_reg_train, acc_class_train, acc_reg_train, error_reg_train = train_loop(train_dataloader,
                                                                              model, loss_func_class, loss_func_reg, optimizer)        
        
        total_loss_test, loss_class_test, loss_reg_test, acc_class_test, acc_reg_test, error_reg_test = test_loop(test_dataloader,
                                                                                        model, loss_func_class, loss_func_reg)

        # Append Metrics
        NN_building.append_history_dict(train_history, total_loss=total_loss_train, loss_class=loss_class_train, loss_reg=loss_reg_train, 
                            acc_class=acc_class_train, acc_reg=acc_reg_train, error_reg=error_reg_train)
        NN_building.append_history_dict(test_history, total_loss=total_loss_test, loss_class=loss_class_test, loss_reg=loss_reg_test, 
                            acc_class=acc_class_test, acc_reg=acc_reg_test, error_reg=error_reg_test)
        
        torch.save(model.state_dict(), f'./results/NN/save_parameters/model{model_type}_lags{lags}_ntraps{ntraps}_epoch{t}.pth')

        
    print("Done!")
    
    #generic.play_ending_song()
    #generic.stop_ending_song(2)


    yhat = NN_building.calc_model_output(model, xtest,loss_func_reg)

    torch.save(model.state_dict(), f'./results/NN/save_parameters/model{model_type}_lags{lags}_ntraps{ntraps}_final.pth')
    NN_building.save_model_mlflow(parameters, model, yhat, ytest, test_history, train_history)



if __name__ == '__main__':
    # Parameters

    repeat = 3 # Number of times the model will be trained and tested

    model_type =  'regressor' # 'classifier' or 'regressor' or 'exponential_renato' or 'linear_regressor' or 'logistical'
    use_trap_info = True
    ntraps = 3
    lags = 3
    random_split = True
    test_size = 0.2
    scale = False
    learning_rate =1e-3
    batch_size = 64
    epochs = 10
    input_3d = False

    parameters = {
        'model_type': model_type,
        'use_trap_info': use_trap_info,
        'ntraps': ntraps,
        'lags': lags,
        'random_split': random_split,
        'test_size': test_size,
        'scale': scale,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'input_3d': input_3d  
        }

    metatron = ['classifier' ,'regressor' ,'exponential_renato' ,'linear_regressor' ,'logistical']
    bool_list = [True, False]

    for boolean1 in bool_list:
        parameters['use_trap_info'] = boolean1
        for boolean2 in bool_list:
            parameters['random_split'] = boolean2
            for type in metatron:
                parameters['model_type'] = type
                for i in range(repeat):
                    NN_pipeline(parameters)