import sklearn.metrics
import torch
from torch.utils.data import Dataset
from torch import nn
import pandas as pd
from typing import Tuple
import utils.NN_arquitectures as NN_arquitectures
import numpy as np
import sklearn
import statsmodels.api as sm 
from pygam import LogisticGAM, s, f, te, l 
import pdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, loguniform
from catboost import CatBoostClassifier

def xy_definition(data:pd.DataFrame, parameters:dict,
                  info_cols:list, eggs_cols:list)->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to define the x and y variables according to the model type.

    Parameters:
    data: dataframe with the data
    paramters: dictionary with the parameters of the model. It must contain
        model_type: string with the model type: classifier, regressor, exponential_renato, linear_regressor, logistic, GAM, 
        use_trap_info: flag to use the traps information like days and distances
        add_constant: flag to add a constant to the x variables
    inf_col: list with the name of columns of days, latitude, longitude, mesepid and novos
    eggs_cols: list with the name of columns of eggs

    Returns:
    x: dataframe with the input variables
    y: dataframe with the output variables


    """
    # Define the output variable according to the model type
    # change ovos flag to 0 and 1 instead of -1 and 1
    if parameters['model_type'] in ['logistic', 'Naive']:      
        ovos_flag = data['novos'].apply(lambda x: 1 if x > 0 else 0)#.rename('ovos_flag', inplace=True)
    elif parameters['model_type'] in ['classifier', 'regressor', 'exponential_renato', 'linear_regressor', 'pareto','mlp', 'random_forest', 'svm', 
                                      'catboost', 'GAM','logistic_3c', 'Naive_3c',  'random_forest_3c', 'svm_3c', 'catboost_3c']:
        ovos_flag = data['novos'].apply(lambda x: 1 if x > 0 else -1)
    else:
        raise ValueError('Model type not found')   

    # output variable
    if parameters['model_type'] in ['classifier', 'logistic', 'GAM', 'Naive', 'mlp', 'random_forest', 'svm', 'catboost']:
        y = ovos_flag
    elif parameters['model_type'] in ['logistic_3c', 'Naive_3c','random_forest_3c','svm_3c', 'catboost_3c']:
        y = data['3_class']
    elif parameters['model_type'] in ['regressor',  'linear_regressor']:
        y = data['novos']
    elif parameters['model_type'] in ['exponential_renato', 'pareto']:
        y = pd.concat([ovos_flag.rename('ovos_flag', inplace=True),data['novos']],axis=1)
    else:
        raise ValueError('Model type not found')

    # input variables
    if parameters['use_trap_info'] == True:
        x = data[info_cols]
    else:
        x = data[eggs_cols]

    # add constant to input variables
    if parameters['add_constant'] == True:
        x = sm.add_constant(x)
    
    if 'mesepid' in x.columns and parameters['model_type'] not in ['catboost','catboost_3c']:
        x = pd.get_dummies(x, columns=['mesepid'], drop_first=True)
        x = x.astype({col: 'float' for col in x.columns if 'mesepid_' in col})
    return x, y

def transform_data_to_tensor(x_train: np.array, x_test: np.array, y_train: np.array, y_test: np.array, model_type: str, device: str)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transform numpy arrays to tensors and send them to the device

    Parameters:
    x_train: numpy array with the training data
    x_test: numpy array with the test data
    y_train: numpy array with the training data
    y_test: numpy array with the test data
    model_type: classifier, regressor, exponential_renato, linear_regressor, logistic, GAM, Naive, mlp, random_forest, svm, catboost
    device: device to send the tensors

    Returns:
    xtrain: tensor with the training data
    xtest: tensor with the test data
    ytrain: tensor with the training data
    ytest: tensor with the test data    
    """

    if model_type in ['classifier','logistic','GAM','Naive','mlp','random_forest','svm','catboost','logistic_3c','Naive_3c','random_forest_3c','svm_3c','catboost_3c']:
        output_type = torch.long
    elif model_type  in ['regressor' , 'exponential_renato' , 'linear_regressor' , 'pareto']:
        output_type = torch.float32

    if isinstance(x_train, pd.DataFrame):
        x_train = x_train.to_numpy()
    if isinstance(x_test, pd.DataFrame):
        x_test = x_test.to_numpy()
    if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    xtrain = torch.tensor(x_train, dtype=torch.float32).to(device)
    xtest = torch.tensor(x_test, dtype=torch.float32).to(device)
    ytrain = torch.tensor(y_train, dtype=output_type).to(device)
    ytest = torch.tensor(y_test, dtype=output_type).to(device)

    return xtrain, xtest, ytrain, ytest

class CustomDataset(Dataset):
    def __init__(self, features, targets, model_type):
        self.features  = features.clone().detach().float()
        if model_type in ['logistic_3c', 'classifier', 'logistic', 'GAM', 'Naive', 'mlp', 'random_forest', 'svm', 'catboost', 'Naive_3c', 'random_forest_3c', 
                          'svm_3c', 'catboost_3c']:
            self.targets =  targets.clone().detach().long()

        elif model_type in ['exponential_renato', 'pareto', 'regressor', 'linear_regressor']:
            self.targets = targets.clone().detach().float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def input_output_sizes(xtrain, model_type):
    # Network structure
    if len(xtrain.shape) == 1:
        model_input = 1
    else:
        model_input = xtrain.shape[1]
        
    if model_type  in ['classifier' , 'exponential_renato' , 'pareto']:# , 'mlp':
        model_output = 2
    elif model_type  in ['regressor', 'linear_regressor', 'logistic', 'GAM', 'Naive', 'mlp', 'random_forest', 'svm', 'catboost', 
                         'logistic_3c', 'Naive_3c', 'random_forest_3c', 'svm_3c', 'catboost_3c']:
        model_output = 1
    return model_input, model_output

def define_model(model_type, model_input, model_output, input_3d,device):
    """
    Function to define pytorch models
    """
    if model_type  in ['linear_regressor']:
        model = NN_arquitectures.LogisticRegression(model_input,input_3d, model_type).to(device)
    elif model_type  in ['exponential_renato']:
        model = NN_arquitectures.NeuralNetworkExponential(model_input, model_output,model_type,input_3d).to(device)
    elif model_type  in ['pareto']:
        model = NN_arquitectures.NeuralNetworkPareto(model_input, model_output,model_type,input_3d).to(device)
    elif model_type  in ['classifier']:
        model = NN_arquitectures.NeuralNetwork(model_input, model_output,model_type,input_3d).to(device)
    elif model_type  in ['mlp']:
        model = NN_arquitectures.mlp(model_input, model_output,model_type,input_3d).to(device)
    else:
        raise ValueError('Model type not found')
    return model

def define_loss_functions(model_type):
    if model_type  in ['classifier' , 'mlp']:
        loss_func_class = nn.CrossEntropyLoss()
        loss_func_reg = None
    elif model_type in ['logistic','GAM','Naive','random_forest', 'svm','catboost','logistic_3c','Naive_3c','random_f,est_3c', 'svm_3c','catboost_3c']:
        loss_func_class = None
        loss_func_reg = None
    elif model_type  in ['regressor' , 'linear_regressor']:
        loss_func_class = None
        loss_func_reg = nn.MSELoss()
    elif model_type  in ['exponential_renato']:
        loss_func_class = nn.CrossEntropyLoss()
        loss_func_reg = NN_arquitectures.ExponentialLoss()
    elif model_type  in ['pareto']:
        loss_func_class = nn.CrossEntropyLoss()
        loss_func_reg = NN_arquitectures.ParetoLoss()

    return loss_func_class, loss_func_reg

def torch_accuracy(yref, yhat,model_type):
    if model_type  in ['classifier' , 'mlp']: #TODO check if this is correct
        return (yhat.argmax(1) == yref).type(torch.float).sum().item()
    elif model_type  in ['regressor' , 'linear_regressor']:
        return ((torch.round(yhat) == yref).type(torch.float)).sum().item()
    else:
        raise ValueError('Model type not found')

def create_history_dict():
    return {
        'total_loss': [],
        'loss_class': [],
        'loss_reg': [],
        'acc_class': [],
        'acc_reg': [],
        'error_reg':[]
    }

def append_history_dict(history_dict, results):
    total_loss, loss_class, loss_reg, acc_class, acc_reg, error_reg = results
    history_dict['total_loss'].append(total_loss)
    history_dict['loss_class'].append(loss_class)
    history_dict['loss_reg'].append(loss_reg)
    history_dict['acc_class'].append(acc_class)
    history_dict['acc_reg'].append(acc_reg)
    history_dict['error_reg'].append(error_reg)

def calc_model_output(model, xtest,loss_func_reg=None):

    if model.model_type  in ['classifier' ,'mlp']:
        yhat = model(xtest).argmax(1).cpu().numpy()
        return yhat
    elif model.model_type  in ['regressor' ,'linear_regressor']:
        yhat = model(xtest).round().cpu().detach().numpy() 
        return yhat.squeeze()
    elif model.model_type  in ['exponential_renato']:
        logit, lamb = model(xtest)
        yhat_reg = 1/lamb
        yhat_class = logit.argmax(1)
        yhat = torch.stack((yhat_class.unsqueeze(1), yhat_reg) ,dim=1)
        return yhat
    elif model.model_type  in ['pareto']:
        logit, alpha = model(xtest)
        x_m = 1 # TODO x_m must be passed as a parameter
        
        yhat_class = logit.argmax(1)
        yhat_reg = yhat_class
        for i in range(len(yhat_reg)):
            if alpha[i]> 1:
                yhat_reg[i] = alpha[i]*x_m/(alpha[i] - 1)*yhat_reg[i]
            else:
                yhat_reg[i] = 1000000*yhat_reg[i]
        yhat = torch.stack((yhat_class, yhat_reg) ,dim=1)
        return yhat
    else:
        raise ValueError('Model type not found')
              
def forward_stepwise(X, y,current_features ,parameters):
    """
    Perform forward stepwise selection to select the best features for a logistic regression model.
    The function starts with an empty set of features and adds the feature that minimizes the AIC at each step.
    This process is repeated until the AIC stops decreasing.

    Parameters:
    X: pd.DataFrame with the input features
    y: pd.Series with the target variable
    current_features: list with the current features used in the model
    parameters: dictionary with the parameters of the model

    Returns:
    final_model: the final logistic regression model
    current_features: list with the selected features
    """
    if parameters['model_type'] in ['logistic']:
        best_aic = sm.Logit(y, X[current_features]).fit().aic
    remaining_features = list(set(X.columns) - set(current_features))

    while remaining_features:
        aic_values = []
        for feature in remaining_features:
            combined_features = current_features + [feature]

            if parameters['model_type'] in ['logistic']:
                model = sm.Logit(y, X[combined_features]).fit()
            elif parameters['model_type'] in ['GAM']:
                spline_terms = []

                if 'semepi' in combined_features:
                    spline_terms.append(s('semepi'))
                if 'latitude0' in combined_features and 'longitude0' in combined_features:
                    spline_terms.append(te('latitude0', 'longitude0'))
                else:
                    if 'latitude0' in combined_features:
                        spline_terms.append(s('latitude0'))
                    if 'longitude0' in combined_features:
                        spline_terms.append(s('longitude0'))
                if spline_terms:
                    model = LogisticGAM(spline_terms + [f for f in combined_features if f not in ['latitude0','longitude0','semepi']])
                    model.gridsearch(X[combined_features], y)
                else:
                    model = LogisticGAM().fit(X[combined_features], y)
                    model.fit(X[combined_features], y)        
            else:
                raise TypeError(f"Model type {parameters['model_type']} not found")
            
            aic_values.append((feature, model.aic))
        
        # Select the feature with the lowest AIC
        best_feature, best_model_aic = min(aic_values, key=lambda x: x[1])
        
        if best_model_aic < best_aic:
            best_aic = best_model_aic
            current_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break
    
    # Fit the final model with selected features
    if parameters['model_type'] in ['logistic']:
        final_model = sm.Logit(y, X[current_features]).fit()
    elif parameters['model_type'] in ['GAM']:
        spline_terms = []
        if 'semepi' in current_features:
            spline_terms.append(s('semepi'))
        if 'latitude0' in current_features and 'longitude0' in current_features:
            spline_terms.append(te('latitude0', 'longitude0'))
        else:
            if 'latitude0' in current_features:
                spline_terms.append(s('latitude0'))
            if 'longitude0' in current_features:
                spline_terms.append(s('longitude0'))
        if spline_terms:
            model = LogisticGAM(spline_terms + [f for f in current_features if f not in ['latitude0','longitude0','semepi']])
            model.gridsearch(X[current_features], y)
        else:
            model = LogisticGAM().fit(X[current_features], y)
            model.fit(X[current_features], y)
    return final_model, current_features

def backward_stepwise(X, y,current_features, parameters):
    """
    Perform backward stepwise selection to select the best features for a logistic regression model.
    The function fits a logistic regression model with all features, then removes the feature with the highest AIC.
    This process is repeated until the AIC stops decreasing.

    Parameters:
    X: pd.DataFrame with the input features
    y: pd.Series with the target variable
    current_features: list with the current features used in the model
    parameters: dictionary with the parameters of the model

    Returns:
    final_model: the final logistic regression model
    current_features: list with the selected features
    """

    if parameters['model_type'] in ['logistic']:
        best_aic = sm.Logit(y, X[current_features]).fit().aic
    remaining_features = list(set(X.columns) - set(current_features))

    while len(current_features) > 0:
        aic_values = []
        for feature in current_features:
            remaining_features = current_features.copy()
            remaining_features.remove(feature)
            if parameters['model_type'] in ['logistic']:
                model = sm.Logit(y, X[remaining_features]).fit()

            elif parameters['model_type'] in ['GAM']:
                spline_terms = []

                if 'semepi' in remaining_features:
                    spline_terms.append(s('semepi'))
                if 'latitude0' in remaining_features and 'longitude0' in remaining_features:
                    spline_terms.append(te('latitude0', 'longitude0'))
                else:
                    if 'latitude0' in remaining_features:
                        spline_terms.append(s('latitude0'))
                    if 'longitude0' in remaining_features:
                        spline_terms.append(s('longitude0'))
                if spline_terms:
                    model = LogisticGAM(spline_terms + [f for f in remaining_features if f not in ['latitude0','longitude0','semepi']])
                    model.gridsearch(X[remaining_features], y)
                else:
                    model = LogisticGAM().fit(X[remaining_features], y)
                    model.fit(X[remaining_features], y)        

            else:
                raise TypeError(f"Model type {parameters['model_type']} not found")
            aic_values.append((feature, model.aic))
        
        # Remove the feature with the highest AIC
        worst_feature, worst_model_aic = max(aic_values, key=lambda x: x[1])
        
        if worst_model_aic < best_aic:
            best_aic = worst_model_aic
            current_features.remove(worst_feature)
        else:
            break
    
    # Fit the final model with selected features
    if parameters['model_type'] in ['logistic']:
        final_model = sm.Logit(y, X[current_features]).fit()
    elif parameters['model_type'] in ['GAM']: 
        spline_terms = []
        if 'semepi' in current_features:
            spline_terms.append(s('semepi'))
        if 'latitude0' in current_features and 'longitude0' in current_features:
            spline_terms.append(te('latitude0', 'longitude0'))
        else:
            if 'latitude0' in current_features:
                spline_terms.append(s('latitude0'))
            if 'longitude0' in current_features:
                spline_terms.append(s('longitude0'))
        if spline_terms:
            model = LogisticGAM(spline_terms + [f for f in current_features if f not in ['latitude0','longitude0','semepi']])
            model.gridsearch(X[current_features], y)
        else:
            model = LogisticGAM().fit(X[current_features], y)
            model.fit(X[current_features], y) 

    return final_model, current_features

def select_model_stepwise(x_train:pd.DataFrame, y_train:pd.DataFrame,parameters:dict, type = 'bidirectional')->Tuple[sm.Logit, list]:
    """
    Select the best model using forward and backward stepwise selection.
    The function compares the AIC of the models obtained with forward and backward stepwise selection
    and returns the model with the lowest AIC.

    Parameters:
    x_train: pd.DataFrame with the input features
    y_train: pd.Series with the target variable
    parameters: dictionary with the parameters of the model
    type: string with the type of stepwise selection: forward, backward or bidirectional

    Returns:
    model: the selected logistic regression model
    selected_features: list with the selected features
    """
    if type == 'forward':
        model_forward, selected_features = forward_stepwise(x_train, y_train, ['const'], parameters)
        return model_forward, selected_features
    elif type == 'backward':
        model_backward, selected_features = backward_stepwise(x_train, y_train, x_train.columns.to_list() ,parameters)
        return model_backward, selected_features
    elif type == 'bidirectional':
        current_features = ['const']
        old_features = []   
        while current_features != old_features:
            old_features = current_features
            _, current_features = forward_stepwise(x_train, y_train,current_features,parameters)
            model_backward, current_features = backward_stepwise(x_train, y_train,current_features,parameters)
        if 'const' not in current_features:
            current_features = ['const'] + current_features 
        return model_backward, current_features

    


def GAM_model(x_train:pd.DataFrame, y_train:pd.DataFrame)->LogisticGAM:
    """
    Fit a Generalized Additive Model (GAM) to the data. The function fits a GAM model with splines for the semepi, latitude and longitude variables.

    Parameters:
    x_train: pd.DataFrame with the input features
    y_train: pd.Series with the target variable
    parameters: dictionary with the parameters of the model

    Returns:
    model: the fitted GAM model
    features: list with the selected features
    """

    features = x_train.columns.tolist()
    spline_terms = []
    if 'semepi' in  features:
        spline_terms.append(s(x_train.columns.get_loc('semepi')))
    if 'latitude0' in   features and 'longitude0' in  features:
        spline_terms.append(te(x_train.columns.get_loc('latitude0'), x_train.columns.get_loc('longitude0')))
    else:
        if 'latitude0' in features:
            spline_terms.append(s(x_train.columns.get_loc('latitude0')))
        if 'longitude0' in  features:
            spline_terms.append(s(x_train.columns.get_loc('longitude0')))
    if spline_terms:
        spline_terms = spline_terms + [l(x_train.columns.get_loc(f)) for f in features if f not in ['latitude0','longitude0','semepi']]
        terms = spline_terms[0]
        for i in spline_terms[1:]:
            terms += i
        model = LogisticGAM(terms).fit(x_train[features], y_train)
    else:
        model = LogisticGAM().fit(x_train[features], y_train)
        model.fit(x_train[features], y_train)        
    return model, features
 

def random_forest_model(x_train:pd.DataFrame, y_train:pd.DataFrame, parameters:dict)->RandomForestClassifier:
    """
    Fit a Random Forest model to the data 

    Parameters:
    x_train: pd.DataFrame with the input features
    y_train: pd.Series with the target variable
    parameters: dictionary with the parameters of the model

    Returns:
    model: the fitted Random Forest model
    """


    if parameters['rf_params']['grid_search'] == True:
            # Initialize the Random Forest model
        rf = RandomForestClassifier()

        # Define the parameter grid
        param_dist = {
            'n_estimators': randint(10, 200),  # Random integer between 50 and 200
            'max_depth': [None, 10, 20, 30, 40],  # Fixed choices
            'min_samples_split': randint(2, 100),  # Random integer between 2 and 10
            'min_samples_leaf': randint(1, 50),  # Random integer between 1 and 5
            'bootstrap': [True, False]  # Binary choice
        }

        grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                        n_iter=20, cv=3, verbose=2, n_jobs=-1)


        # Fit the model to the training data
        grid_search.fit(x_train, y_train)
        model = grid_search.best_estimator_
        parameters['rf_params'] = grid_search.best_params_  
        parameters['rf_params']['grid_search'] = True


    else:
        model = RandomForestClassifier(
                                        n_estimators=int(parameters['rf_params']["n_estimators"]),
                                        max_depth=int(parameters['rf_params']["max_depth"]),
                                        min_samples_split=int(parameters['rf_params']["min_samples_split"]),
                                        min_samples_leaf=int(parameters['rf_params']["min_samples_leaf"]),
                                        bootstrap=parameters['rf_params']["bootstrap"]
                                    )
        model.fit(x_train, y_train)
    return model, parameters


def svm_model(x_train:pd.DataFrame, y_train:pd.DataFrame, x_test:pd.DataFrame, y_test:pd.DataFrame, parameters:dict)->RandomForestClassifier:
    """
    Fit a Support Vector Machine (SVM) model to the data.

    Parameters:
    x_train: pd.DataFrame with the input features
    y_train: pd.Series with the target variable
    parameters: dictionary with the parameters of the model

    Returns:
    model: the fitted SVM model
    """

    if parameters['svm_params']['grid_search'] == True:
        
        from sklearn.pipeline import Pipeline
        from sklearn.kernel_approximation import RBFSampler
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import SGDClassifier
        from sklearn.model_selection import ParameterSampler
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.metrics import accuracy_score
        


        # Define the custom TransformerSelector class
        class TransformerSelector(BaseEstimator, TransformerMixin):
            """Custom transformer to select between RBF, Polynomial, or None."""
            def __init__(self, transformer=None):
                self.transformer = transformer

            def fit(self, X, y=None):
                if self.transformer is not None:
                    self.transformer.fit(X, y)
                return self

            def transform(self, X):
                if self.transformer is not None:
                    return self.transformer.transform(X)
                return X

            def fit_transform(self, X, y=None):
                self.fit(X, y)
                return self.transform(X)

        # Define the pipeline
        pipeline = Pipeline([
            ('transformer', TransformerSelector()),  # Placeholder for transformations
            ('sgd', SGDClassifier(max_iter=1000, tol=1e-3))
        ])

        # Define the parameter grid
        param_dist = {
            # Transformation selection
            'transformer': [
                None,  # No transformation
                RBFSampler(),  # RBF Kernel approximation
                #PolynomialFeatures()  # Polynomial projection
            ],
            # Parameters for RBF Kernel
            'transformer__gamma': [0.1, 0.5, 1.0, 2.0],
            # Parameters for PolynomialFeatures
            #'transformer__degree': [2, 3, 4, 5],
            #'transformer__interaction_only': [False],
            # SGDClassifier parameters
            'sgd__alpha': np.logspace(-5, -1, 10),
            'sgd__penalty': ['l2', 'l1', 'elasticnet'],
            'sgd__l1_ratio': np.linspace(0.1, 1, 5)
        }

        # Custom evaluation function
        def evaluate_model(pipeline, params, x_train, y_train, x_test, y_test, n_runs=1):
            scores = []
            best_model = None
            best_score = 0 
            for _ in range(n_runs):
                pipeline.set_params(**params)  # Update pipeline with current parameters
                pipeline.fit(x_train, y_train)  # Fit the model
                y_pred = pipeline.predict(x_test)  # Predict
                scores.append(accuracy_score(y_test, y_pred))  # Compute accuracy
                if scores[-1] > best_score:
                    best_score = scores[-1]
                    best_model = pipeline
            return np.mean(scores), best_model

        # Generate parameter combinations
        n_iter = 10
        param_list = list(ParameterSampler(param_dist, n_iter=n_iter))

        # Run the search
        results = []
        models= []
        for i, params in enumerate(param_list):
            print(f"Evaluating parameter set {i + 1}/{len(param_list)}...")
            print(params)
            # Filter invalid combinations (e.g., gamma for non-RBF transformers)
            if params['transformer'] is None:
                params = {k: v for k, v in params.items() if not k.startswith('transformer')}
            else:
                if isinstance(params['transformer'], PolynomialFeatures):
                    # Remove `gamma` for PolynomialFeatures
                    params = {k: v for k, v in params.items() if k not in ['transformer__gamma']}
                # Filter invalid combinations (e.g., degree for non-Polynomial transformers)
                if isinstance(params['transformer'], RBFSampler):
                    # Remove `degree` for RBF Sampler
                    params = {k: v for k, v in params.items() if k not in ['transformer__degree', 'transformer__interaction_only']}

            mean_score, model = evaluate_model(pipeline, params, x_train, y_train, x_test, y_test, n_runs=10)
            results.append({'params': params, 'mean_score': mean_score})
            models.append(model)    

        # Select the best model
        results_df = pd.DataFrame(results)
        best_result = results_df.loc[results_df['mean_score'].idxmax()]
        model = models[results_df['mean_score'].idxmax()]

        # Save best model parameters 
        if 'transformer' in best_result['params'].keys():
            parameters['svm_params']['gamma'] = best_result['params']['transformer__gamma']
            parameters['svm_params']['kernel'] = 'rbf'
        else:
            parameters['svm_params']['gamma'] = 'None'
            parameters['svm_params']['kernel'] = 'linear'
        
        
        parameters['svm_params']['alpha'] = best_result['params']['sgd__alpha']
        parameters['svm_params']['penalty'] = best_result['params']['sgd__penalty']
        parameters['svm_params']['l1_ratio'] = best_result['params']['sgd__l1_ratio']

    else:
        model = SVC( C=1/parameters['svm_params']['alpha'],
                        gamma=parameters['svm_params']['gamma'],
                        kernel=parameters['svm_params']['kernel'],
                        degree=parameters['svm_params']['degree'],
                        class_weight='balanced'
                        
                        )
        model.fit(x_train, y_train)
    return model,parameters

def catboost_model(x_train:pd.DataFrame, y_train:pd.DataFrame,parameters:dict)->CatBoostClassifier:
    """
    Fit a CatBoost model to the data.

    Parameters:
    x_train: pd.DataFrame with the input features
    y_train: pd.Series with the target variable
    parameters: dictionary with the parameters of the model

    Returns:
    model: the fitted CatBoost model
    """
    if parameters['catboost_params']['grid_search'] == True:
        # Define the parameter grid
        param_dist = {
            'iterations': randint(100, 500),  # Random integer between 100 and 500
            'learning_rate': np.linspace(0.001, 0.2, 100),  # 0.01 to 0.2
            'depth': randint(1,20),
            'l2_leaf_reg':  randint(1,10),
            'random_strength':  randint(1,10),

        }

        grid_search = RandomizedSearchCV(estimator=CatBoostClassifier(), param_distributions=param_dist,
                                        n_iter=20, cv=3, verbose=2, n_jobs=-1)


        # Fit the model to the training data
        grid_search.fit(x_train, y_train)
        model = grid_search.best_estimator_
        parameters['catboost_params'] = grid_search.best_params_  
        parameters['catboost_params']['grid_search'] = True
    else:
        # Train a CatBoost model
        model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0)
        model.fit(x_train, y_train,cat_features=['mesepid'])

    return model,parameters


def easy_save(train_history:dict, test_history:dict, yhat_train:list, ytrain:list, yhat:list, ytest:list, model_type:str,loss_func_class:float, loss_func_reg:float):
    """
    Function to save the results of the model in a dictionary previously created  
    """
    
    results_train = evaluate_NN(model_type,loss_func_class, loss_func_reg, yhat_train, ytrain)
    results_test = evaluate_NN(model_type,loss_func_class, loss_func_reg, yhat, ytest)
    append_history_dict(train_history, results_train)
    append_history_dict(test_history, results_test)
    return train_history, test_history  

def evaluate_NN(model_type,loss_func_class, loss_func_reg, yhat, y ):

    if model_type in['logistic', 'GAM', 'Naive', 'mlp', 'random_forest', 'svm', 'catboost', 'logistic_3c', 'Naive_3c', 'random_forest_3c', 'svm_3c', 'catboost_3c']:

        loss_reg =  0
        loss_class =  0
        acc_class = sklearn.metrics.accuracy_score(y, yhat)
        acc_reg =  0 
        error_reg =  0
        total_loss = 0
        return total_loss, loss_class, loss_reg, acc_class, acc_reg, error_reg

    elif model_type in ['classifier']:# or 'mlp':
        loss_class = loss_func_class(yhat.squeeze(), y) # yhat = logit
        loss_reg =  torch.tensor(0)
        acc_class = torch_accuracy(y,  yhat, model_type)
        acc_reg =  0
        error_reg =  0

    elif model_type in ['regressor' , 'linear_regressor']:
        yhat = yhat.squeeze()
        loss_class =  torch.tensor(0)
        loss_reg = loss_func_reg(yhat, y)
        acc_class =  0
        acc_reg = torch_accuracy(y, yhat, model_type)
        error_reg = ((yhat - y)**2).sum().item()
    
    elif model_type in ['exponential_renato']:
        logit, lamb = yhat
        # evaluate the pdf of the distribution
        
        yhat = torch.round(1/(lamb + 0.00001)) # rounded mean of the distribution
        # split NN output
        y_class =  y[:,0].long()
        y_reg = y[:,1]
        # calculate metrics
        loss_class = loss_func_class(logit, y_class)
        loss_reg = loss_func_reg(y_reg, lamb)
        acc_class = torch_accuracy(y_class, logit, 'classifier')
        acc_reg = torch_accuracy(y_reg, yhat, 'regressor')
        error_reg = ((yhat - y_reg)**2).sum().item() #mse

    elif model_type in ['pareto']:

        logit, alpha = yhat
        x_m = 1 # TODO: if x_m must be implemented as a parameter, it must be passed in yhat 
        # evaluate the pdf of the distribution
        yhat = alpha*x_m/(alpha - 1) 

        y_class =  y[:,0].long()
        y_reg = y[:,1]
        # calculate metrics
        loss_class = loss_func_class(logit, y_class)
        loss_reg = loss_func_reg(y_reg, alpha)
        acc_class = torch_accuracy(y_class, logit, 'classifier')
        acc_reg = torch_accuracy(y_reg, yhat, 'regressor')
        error_reg = ((yhat - y_reg)**2).sum().item() #mse
    else:   
        raise ValueError('Model type not found')


    return loss_class, loss_reg, acc_class, acc_reg, error_reg

