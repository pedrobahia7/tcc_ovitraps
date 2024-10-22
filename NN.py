import utils.NN_building as NN_building

# Parameters

model_type =  'regressor' # 'classifier' or 'regressor' or 'exponential_renato'
use_trap_info = True
ntraps = 3
lags = 3
random_split = False
test_size = 0.2
scale = False
learning_rate =1e-3
batch_size = 64
epochs = 1



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
    'epochs': epochs   
    
    }
NN_building.NN_creation(parameters)