from geopy.distance import geodesic
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
import pdb



def get_valid_samples(df:pd.DataFrame)->pd.DataFrame:
    """
    Function to get the samples that have the coordinates in the dataframe.

    Parameters:
    df: pandas dataframe

    Returns:
    pandas dataframe
    
    """
    return df[['nplaca','novos','latitude','longitude','narmad','ano','semepi','dtcol']].drop_duplicates().dropna().reset_index(drop=True)


def same_coord_samples(df:pd.DataFrame)->pd.DataFrame:
    """
    Get the samples that have the coordinates in the dataframe.

    Parameters:
    df: pandas dataframe

    Returns:
    pandas dataframe
    """
    
    groups = df.groupby(['latitude','longitude'])
    rows_to_update = []

    for _, group in groups:
        if group['narmad'].nunique() > 1:  # More than one unique value in C
            rows_to_update.extend(group.index.tolist())  # Add row indices to the list
    return df.loc[rows_to_update]


def create_distance_matrix(position_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create a distance matrix from a position matrix of traps using geodesic distances.

    Parameters:
    position_matrix (pd.DataFrame): DataFrame with columns 'latitude', 'longitude' and 'narmad'

    Returns:
    distance_matrix (pd.DataFrame): DataFrame with the distances between each trap.
    nbg  Trap numbers are used as index and columns
    """

    coordinates = position_matrix[['latitude', 'longitude']].to_numpy()

    distance_list = []
    for idx1,idx2 in tqdm(itertools.combinations(range(len(coordinates)), 2)):
        distance_list.append(geodesic(coordinates[idx1], coordinates[idx2]).meters)

    # Size of the matrix
    n = len(coordinates)

    # Create an empty square matrix
    symmetric_matrix = np.zeros((n, n))

    # Fill the upper triangular part (including the diagonal)
    symmetric_matrix[np.triu_indices(n, 1)] = np.array(distance_list)

    # Fill the lower triangular part by mirroring the upper part
    symmetric_matrix += symmetric_matrix.T

    # Convert the NumPy array to a DataFrame
    distance_matrix = pd.DataFrame(symmetric_matrix, index=position_matrix.index, columns=position_matrix.index)

    # Set index and columns to 'narmad' values
    distance_matrix.columns = position_matrix['narmad'].values

    return distance_matrix


def get_distance_matrix(df:str,address:str = './results/distance_matrix.csv')->pd.DataFrame:
    """
    Function to laod the distance matrix. If it doesn't exist calculate if
    from the coordinates of the traps

    Parameters:
    df: pandas dataframe with the data in Dilermando format
    address: string with the address of the distance matrix file

    Returns:
    distance_matrix (pd.DataFrame): DataFrame with the distances between each trap.
    """


    valid_samples = get_valid_samples(df)
    position_matrix = valid_samples[['latitude','longitude','narmad']].drop_duplicates().dropna().reset_index(drop=True)

    try: 
        distance_matrix = pd.read_csv(address,index_col=0)
        distance_matrix.columns = distance_matrix.columns.astype(float)
        distance_matrix.set_index(distance_matrix.columns,inplace=True)


    except:
        distance_matrix = create_distance_matrix(position_matrix)
        distance_matrix.to_csv(address)
    return distance_matrix


def create_week_trap_df(df:pd.DataFrame)->pd.DataFrame:
    """
    Create a dataframe of the number of eggs(value), in each week(rows) in each trap(columns)

    Parameters:
    df: pandas dataframe with the data in Dilermando format

    Returns:
    week_trap_df: pandas dataframe with the number of eggs in each week in each trap
    
    """

    valid_samples = get_valid_samples(df)
    week_trap_df = valid_samples.pivot(index=['ano','semepi'],columns='narmad',values='novos')
    new_index = pd.MultiIndex.from_product([week_trap_df.index.levels[0], range(101,153)]) # introduce weeks 51 and 52
    new_index = new_index[(new_index <= week_trap_df.iloc[-1].name) & (new_index >= week_trap_df.iloc[0].name)] #remove indexes that are greater than the last sample or smaller than the first one
    week_trap_df = week_trap_df.reindex(new_index) # [week,trap] - > novos
    return week_trap_df


def create_lagged_eggs(df:pd.DataFrame,lags:int)->np.ndarray:
    """
    Create a  numpy array from a dataframe with the number lags defined as a paramenters
    [lag x week x trap] -> df[value]. 
    Obs.: considers 2*lags due to the biweekly sampling rate
    
    Parameters:
    df: pandas dataframe with the number of eggs in each week in each trap
    lags: number of lags to be created

    Returns:
    lagged_eggs: numpy array with the lagged matrix
    """
    list_lagged_df = []
    for i in range(1,2*lags+1):
        list_lagged_df.append(df.shift(i).values)
    lagged_eggs = np.stack(list_lagged_df, axis=0) 
    return lagged_eggs


def convert_dates_to_ordinal(df:pd.DataFrame)->pd.DataFrame:
    """
    Create numpy array of an ordinal number representing each day, maintaining the difference property
    [lag x week x trap] -> ordinal days.
    Obs.: consider 2*lags as the dimension due to the biweekly sampling rate 

    Parameters:
    df: pandas dataframe with the data in Dilermando format containing valid samples
    lags: number of lags to be created

    Returns:
    week_trap_df: pandas dataframe with ordinal numbers representing days
    
    """

    day_df = df.pivot(index=['ano','semepi'],columns='narmad',values='dtcol')
    new_index = pd.MultiIndex.from_product([day_df.index.levels[0], range(101,153)]) # introduce weeks 51 and 52
    new_index = new_index[(new_index <= day_df.iloc[-1].name) & (new_index >= day_df.iloc[0].name)] #remove indexes that are greater than the last sample or smaller than the first one
    day_df = day_df.reindex(new_index) # [week,trap] - > dtcol
    day_df = day_df.map(lambda x: x.toordinal() if pd.notnull(x) else np.nan) # convert to ordinal so we can calculate the difference between two dates
        
    return day_df


def final_matrix_logic(valid_samples:pd.DataFrame, day_df:pd.DataFrame, distance_matrix:np.array, nan_count_matrix:np.array, lagged_eggs:np.array, 
                        lagged_days:np.array, info_df:dict, trap_index_dict:dict, index_trap_dict:dict, nplaca_index_dict:dict, lags:int, n_traps:int, nplaca_lat_dict:dict,
                        nplaca_long_dict:dict)->pd.DataFrame:
    """
    Function to create the final matrix to be used in the neural network model

    Parameters:
    valid_samples: pandas dataframe with the valid samples
    day_df: pandas dataframe with the ordinal days
    distance_matrix: pandas dataframe with the distance between the traps
    nan_count_matrix: numpy array with the number of autoregressive traps that have nan for each trap
    lagged_eggs: numpy array with the lagged number of eggs
    lagged_days: numpy array with the lagged days in ordinal 
    info_df: pandas dataframe with the information of each trap. It must contain the columns 'narmad' and 'nplaca' 
    trap_index_dict: dictionary with the index of each trap in the egg matrix and distance matrix
    index_trap_dict: dictionary with the trap of each index in the egg matrix
    nplaca_index_dict: dictionary with the index of each placa in the egg matrix
    lags: number of lags to be created
    n_traps: number of traps to be considered, including the original trap
    nplaca_lat_dict: dictionary with the latitude of each placa
    nplaca_long_dict: dictionary with the longitude of each placa


    Returns:
    final_df: pandas dataframe with the final matrix to be used in the neural network model
    """
    #convert dfs to numpy
    distance_matrix_np = distance_matrix.to_numpy()
    day_df_np = day_df.to_numpy()
    
    #create lists to store the final samples
    list_placas = []
    list_final_samples = []

    #iterate over the traps
    for original_trap in distance_matrix.columns:
        matching_placas = info_df[info_df['narmad'] == original_trap]['nplaca']                     # get the placas of the trap
        original_trap_index = trap_index_dict[original_trap]                                        # get the index of the eggs matrix referent to the trap
        sorted_distance_indexes = np.argsort(distance_matrix_np[original_trap_index])               # sort the other traps by distance from the trap
        if sorted_distance_indexes[0] != original_trap_index:
            print('trap is not the closest to itself')
        assert sorted_distance_indexes[0] == original_trap_index, 'trap is not the closest to itself'
        # iterate over placas of the trap
        for placa in matching_placas:  
            add_row = [placa]                                                                       # create a row to add to the final dataframe
            order_index = 0                                                                         # index of the sorted traps
            list_placas.append(placa)                                                               # add the placa to a list to check if there are duplicates
            week_index = nplaca_index_dict[placa]                                                   # get the index of the eggs matrix referring to the week

            
            if nan_count_matrix[week_index,original_trap_index] > lags:                             # remove samples of original traps that doesn't have enough data
                continue                                                                            # autoregressive samples 
               
            #iterate over the neighbors closest to the original trap
            for _ in range(n_traps): 
                neighbor_index = sorted_distance_indexes[order_index]                                    # get the index of the eggs matrix referent to the neighbor trap
                
                # loop to avoid traps that don't have enough autoregressive samples
                while nan_count_matrix[week_index,neighbor_index] > lags:                               
                    order_index += 1
                    if order_index >= len(sorted_distance_indexes):                                     # if there are no more neighbors to add
                        break
                    neighbor_index = sorted_distance_indexes[order_index]
                    #if order_index > 50: #avoid arbitrarily distant traps #it was not necessary
                        #break
                if order_index >= len(sorted_distance_indexes):                                     # if there are no more neighbors to add
                        break
                lagged_samples = lagged_eggs[:,week_index,neighbor_index]                               # get the eggs of all the lags. [lag x week x trap] -> novos

                [add_row.append(i) for i in lagged_samples[~np.isnan(lagged_samples)]]              # add lagged eggs
                add_row.append(nplaca_lat_dict[index_trap_dict[neighbor_index]])            # add latitude of the second trap
                add_row.append(nplaca_long_dict[index_trap_dict[neighbor_index]])           # add longitude of the second trap

                
                #subtract lagged days from orignal sample day [lag x week x trap] -> ordinal days
                lagged_samples_days = lagged_days[:,week_index,neighbor_index]
                days_diff = day_df_np[week_index,original_trap_index] - lagged_samples_days[~np.isnan(lagged_samples_days)] 
                
                [add_row.append(i) for i in days_diff]                                              # add lagged days
                order_index += 1
            if order_index >= len(sorted_distance_indexes):
                        break
            list_final_samples.append(add_row)


    #assert len(list_placas) == valid_samples.shape[0], 'invalid number of placas'
    assert len(list_placas) == len(set(list_placas)), 'duplicated placas'

    # create the final dataframe
    final_df = pd.DataFrame(list_final_samples)                             

    #create the columns names
    columns_names = ['nplaca']
    for j in range(n_traps): 
        for i in range(1,lags+1):
            columns_names.extend(['trap'+str(j)+'_lag'+str(i)])
        columns_names.extend(['latitude'+str(j)])
        columns_names.extend(['longitude'+str(j)])
        for i in range(1,lags+1):
            columns_names.extend(['days'+str(j)+'_lag'+str(i)])
    final_df.columns = columns_names  

    #add the number of eggs in the current week (output)
    final_df = pd.merge(valid_samples[['nplaca','novos','dtcol']],final_df,how='inner',on='nplaca') ###
    assert final_df.isna().sum().sum() == 0, 'There are nans in the final dataframe'

    return final_df


def create_final_matrix(lags:str, n_traps:str,save_path:str, data_addr:str = './data/final_data.csv')->pd.DataFrame:
    """
    Function to create the final matrix to be used in the neural network model

    Parameters:
    lags: number of lags to be created
    n_traps: number of traps to be considered, including the original trap
    save_path: string with the address to save the final matrix
    data_addr: string with the address of the data file

    Returns:
    final_df: pandas dataframe with the final matrix to be used in the neural network model
    """
    print('Creating final matrix')
    data = pd.read_csv(data_addr,parse_dates=['dtcol'])
    valid_samples = get_valid_samples(data)

    # introduce a small value on traps with the same coordinates to differentiate them
    same_coord = same_coord_samples(valid_samples)
    for trap in same_coord['narmad'].unique():
        valid_samples.loc[valid_samples['narmad'] == trap, 'latitude'] += np.random.rand()*0.00000001
        valid_samples.loc[valid_samples['narmad'] == trap, 'longitude'] += np.random.rand()*0.00000001

    # Distance matrix
    distance_matrix = get_distance_matrix(valid_samples)

    # Create the week_trap_df
    week_trap_df = create_week_trap_df(valid_samples)

    # Create the lagged eggs
    lagged_eggs = create_lagged_eggs(week_trap_df, lags)

    # Create the nan_count_matrix
    nan_count_matrix = np.sum(np.isnan(lagged_eggs), axis=0) # [week x trap] -> number of nans in the lagged matrix
    for i in range(2*lags):
        print("Number of samples with",i+1,"invalid values: ",np.sum(nan_count_matrix==i+1))

    # info matrix
    info_df = valid_samples[['ano','semepi','nplaca','dtcol','novos','narmad','latitude','longitude']]

    # lagged days matrix
    day_df = convert_dates_to_ordinal(valid_samples)
    lagged_days = create_lagged_eggs(day_df,lags) # [lag x week x trap] -> ordinal days

    #useful dicts   
    trap_index_dict = {trap: index for index,trap in enumerate(distance_matrix.columns)}                                           # trap: index 
    index_trap_dict = {index: trap for index,trap in enumerate(distance_matrix.columns)}                                           # trap: index 
    
    yearweek_index_dict = {(year,week): index for index,(year,week) in enumerate(week_trap_df.index)}                           # (year,week): index
    nplaca_week_dict = {nplaca: (year, week) for nplaca,week,year in zip(info_df['nplaca'],info_df['semepi'],info_df['ano'])}   # nplaca: (year,week)
    nplaca_index_dict = {nplaca: yearweek_index_dict[(year, week)] for nplaca,week,year in zip(info_df['nplaca'],info_df['semepi'],info_df['ano'])}   # nplaca: week index 
    
    
    unique_position = valid_samples[['latitude','longitude']].drop_duplicates().dropna().reset_index(drop=True)
    lat_mean = unique_position['latitude'].mean()
    long_mean = unique_position['longitude'].mean()
    lat_std = unique_position['latitude'].std()
    long_std = unique_position['longitude'].std()
    
    trap_lat_dict = {narmad: (lat - lat_mean)/lat_std for narmad,lat in zip(info_df['narmad'],info_df['latitude'])}                                  # narmad: latitude
    trap_long_dict = {narmad: (long - long_mean)/long_std for narmad,long in zip(info_df['narmad'],info_df['longitude'])}                              # nplaca: longitude

    #final matrix
    final_df = final_matrix_logic(valid_samples, day_df, distance_matrix, nan_count_matrix,
            lagged_eggs, lagged_days, info_df, trap_index_dict, index_trap_dict, nplaca_index_dict, lags, n_traps,trap_lat_dict,trap_long_dict)
    
    # add the week of the principal trap
    info_df['semepi'] = info_df['semepi'] - 100
    final_df = final_df.merge(info_df[['nplaca','semepi']], on='nplaca', how='left') 

    #add zero perc
    zero_perc = pd.DataFrame()
    for trap in distance_matrix.columns:
        trap_df = data[data['narmad'] == trap][['novos','dtcol','nplaca']].sort_values(by='dtcol').reset_index(drop=True)  

        trap_df['is_zero'] = trap_df['novos'] == 0
        trap_df['cumulative_zeros'] = trap_df['is_zero'].cumsum().shift(1)
        trap_df['zero_perc'] = (trap_df['cumulative_zeros']/ (trap_df.index))
        zero_perc = pd.concat([zero_perc,trap_df[['nplaca','zero_perc']]])
    final_df = pd.merge(final_df,zero_perc[['nplaca','zero_perc']],on='nplaca',how='left')

    final_df.sort_values(by=['dtcol'],inplace=True)
    final_df.drop(columns=['dtcol'],inplace=True)
    final_df.reset_index(drop=True)
    final_df.to_parquet(save_path,index=False,compression='snappy')
    return final_df


def data_train_test_split(x, y, test_size, random_split,ovos_flag):
    n = x.shape[0]

    if random_split:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=ovos_flag)
    else:
        train_size = 1 - test_size

        y_train = y.iloc[:int(n*train_size)]
        y_test = y.iloc[int(n*train_size):]
        x_train = x.iloc[:int(n*train_size)]
        x_test = x.iloc[int(n*train_size):]    
    return x_train, x_test, y_train, y_test

def scale_column(x_train:pd.DataFrame, x_test:pd.DataFrame, column:list)->Tuple[pd.DataFrame,pd.DataFrame,int]:
    """
    Scales the same nature columns of x_train and x_test using the MinMaxScaler. The reference column is the one with the maximum value.

    Parameters:
    x_train: pandas dataframe
    x_test: pandas dataframe
    column: list of column names to be scaled

    Returns:
    x_train: pandas dataframe
    x_test: pandas dataframe
    max_value: maximum value of the reference column

    """
    max_value = x_train[column].max().max()
    x_train.loc[:, column] = x_train.loc[:, column] / max_value
    x_test.loc[:, column] = x_test.loc[:, column] / max_value
    return x_train, x_test, max_value



def scale_dataset(x_train, x_test, y_train, y_test, model_type, use_trap_info, eggs_columns, lat_columns,long_columns, days_columns):

    x_train, x_test, max_eggs = scale_column(x_train, x_test, eggs_columns)
    if use_trap_info:
        x_train, x_test, max_days = scale_column(x_train, x_test, days_columns)
    if model_type == 'regressor':
        y_train = y_train/max_eggs
        y_test = y_test/max_eggs
    elif model_type == 'exponential_renato':
        y_train['novos'] = y_train['novos']/max_eggs
        y_test['novos'] = y_test['novos']/max_eggs
    return x_train, x_test, y_train, y_test



def create_3d_input(df, num_traps, num_lags):
    """
    Create a 3D NumPy array from a DataFrame with lagged trap and days data.
    The DataFrame must have columns for each trap and each lag, with the format:
    trap0_lag1, trap0_lag2, ..., trap0_lagN, trap1_lag1, ..., trapN_lagN, distance0, ..., distanceN,
    days0_lag1, ..., daysN_lagN.

    Parameters:
    df: DataFrame with the lagged trap and days data
    num_traps: Number of traps
    num_lags: Number of lags

    Returns:
    result_3d_array: 3D NumPy array with shape (num_weeks, num_traps * num_lags, 3)   
    
    """
    
    num_weeks = df.shape[0]  # Number of rows (weeks)

    # Extract trap lag columns, days lag columns, and distances into NumPy arrays
    trap_lags = np.hstack([df[[f'trap{trap_num}_lag{i}' for i in range(1, num_lags + 1)]].values 
                        for trap_num in range(num_traps)])

    days_lags = np.hstack([df[[f'days{trap_num}_lag{i}' for i in range(1, num_lags + 1)]].values 
                        for trap_num in range(num_traps)])

    distances = np.hstack([df[[f'distance{trap_num}']].values.repeat(num_lags, axis=1) 
                        for trap_num in range(num_traps)])

    # Stack trap lags, days lags, and distances into a 3D array
    # Shape: (num_weeks, num_traps * num_lags, 3)
    result_3d_array = np.stack((trap_lags, days_lags, distances), axis=-1)
    return result_3d_array
















