import pandas as pd
from IPython.display import display
import numpy as np
from typing import Callable, Optional
import matplotlib.pyplot as plt
import pygame
import time
from geopy.distance import distance



def print_rows_with_nan(df:pd.DataFrame,col:str,return_rows = False, print_rows= True)->Optional[pd.DataFrame]:
    """
    Print rows with NaN values in the column col of the dataframe df.

    Parameters:
    df: pandas dataframe
    col: string

    Returns:
    Rows with NaN values in the column col of the dataframe df.
    """
    nan_rows = df[df[col].isna()]   
    
    if print_rows:
        display(nan_rows)
    if return_rows:
        return nan_rows
    
def get_col_types(df:pd.DataFrame,col:str)->pd.DataFrame:
    """
    Get the data types of the columns in the dataframe df.

    Parameters:
    df: pandas dataframe

    Returns:
    pandas dataframe
    """

    return df[col].apply(type).unique()

def get_col_unique_values(df:pd.DataFrame,col:str)->np.ndarray:
    """
    Get the unique values of the columns in the dataframe df.

    Parameters:
    df: pandas dataframe

    Returns:
    pandas dataframe
    """

    return df[col].unique()

def info_dict_col(df:pd.DataFrame, columns:list, func:Callable[pd.DataFrame,str])->dict:
    """
    Returns a dictionary with the information of each column in the dataframe.
    The information is obtained by applying a function to each column

    Parameters:
    df: DataFrame to be analyzed
    columns: list of column names to be analyzed
    func: function that takes a dataframe and a column name as arguments

    Returns:
    info_dict: dict containing the name of each column and the information obtained by applying the function to it


    """
    info_dict = {col:func(df,col) for col in columns}
    return info_dict

def print_info_col(info_dict:dict,limit:int=-1)->None:
    """
    Prints the information of each column in the dictionary

    Parameters:
    info_dict: dict containing the information of each column
    limit: number of elements to be printed; if -1, all elements are printed

    Returns:
    None

    """
    for col,value in info_dict.items():
        print(f'{col} [{value.shape[0]}]: {value[:limit]}') # print name [count]: value

def df_difference(df1:pd.DataFrame, df2:pd.DataFrame)->None:
    """
    Compare two DataFrames and print the differing cells.

    Parameters:
    df1: pandas dataframe
    df2: pandas dataframe

    Returns:
    None
    """
    df1 = df1.dropna()
    df2 = df2.dropna()
    # Compare DataFrames
    diff = df1 != df2
    print('Number of differing cells:', diff.sum().sum())
    # Print differing cells
    for i in range(len(df1)):
        for j in range(len(df1.columns)):
            if diff.iat[i, j]:
                print(f"Difference at row {i}, column {df1.columns[j]}: {df1.iat[i, j]} != {df2.iat[i, j]}")

def row_with_value(df,column,value):
    """
    Get the rows of the dataframe df where the column column has the value.

    Parameters:
    df: pandas dataframe
    column: string
    value: any

    Returns:
    pandas dataframe

    """
    return df.where(df[column] == value).dropna(axis=0, how='all')    



def hist_html(df,trap):
    # Generate a histogram
    plt.figure(figsize=(6, 4))
    plt.hist(df, bins=1000, color='black', edgecolor='black')
    plt.title(f'Histograma de contagem de ovos - armadilha {trap}')
    plt.xlabel('Contagem de ovos')
    plt.ylabel('Frequência')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'./results/histograms/histogram_{trap}.png', bbox_inches='tight', dpi=300)
    plt.close()  # Close the plot to avoid displaying it in the output

    return f'./histograms/histogram_{trap}.png'


def play_finish_song(addr = './data/Sinfonia To Cantata # 29.mp3'):
    # Initialize the mixer
    pygame.mixer.init()
    # Load the MP3 file
    pygame.mixer.music.load(addr)
    # Play the MP3 file
    pygame.mixer.music.play()


def stop_finish_song(seconds = 5):
    time.sleep(seconds)
    pygame.mixer.music.stop()
    # Optional: Clean up the mixer
    pygame.mixer.quit()


def pareto_plot(data:pd.Series,plt_title:str,ax=plt ):
    """
    Generate a log-log plot of the data so that the Pareto distribution can be observed.

    Parameters:
    data: pandas Series
    plt_title: string
    ax: matplotlib.pyplot object

    Returns:
    None
    """
    name = data.name                                        #get name of the column
    df = data.value_counts().sort_index().reset_index()     #count values
    df.drop(df[df[name] == 0].index, inplace=True)          #drop index 0
    df = df.apply(lambda x: np.log(x))                      # apply log to values
    ax.scatter(df[name], df['count'],s=1)
    if ax == plt:
        ax.title(plt_title)
        ax.ylabel(f'Log of frequency')
        ax.xlabel(f'Log of value')

    else:
        ax.set_title(plt_title)
    
    
def pareto_plot_html(df:pd.DataFrame,trap:str) -> str:
    """
    Generate a Pareto plot of the data of a specific trap so it can be saved as an image file and used in the html

    Parameters:
    df: pandas DataFrame with the number of traps
    trap: string refering to the trap number

    Returns:
    str: path to the image file
    """
    # Generate a histogram
    plt.figure(figsize=(6, 4))
    pareto_plot(df, f'Pareto plot - armadilha {trap}')
    plt.savefig(f'./results/pareto_plot/pareto_plot_{trap}.png', bbox_inches='tight', dpi=300)
    plt.close()  # Close the plot to avoid displaying it in the output

    return f'./pareto_plot/pareto_plot_{trap}.png'


def create_distance_matrix(position_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create a distance matrix from a position matrix of traps

    Parameters:
    position_matrix (pd.DataFrame): DataFrame with columns 'latitude', 'longitude' and 'narmad'

    Returns:
    distance_matrix (pd.DataFrame): DataFrame with the distances between each trap.
    nbg  Trap numbers are used as index and columns
    """
    distance_matrix = pd.DataFrame(index=position_matrix.index, columns=position_matrix.index)
    coordinates = position_matrix.apply(lambda row: (row['latitude'], row['longitude']), axis=1)
    for i, coord1 in enumerate(coordinates):
        distance_matrix.iloc[i] = coordinates.apply(lambda coord2: distance(coord1, coord2).meters)
    distance_matrix = distance_matrix.dropna(how='all',axis=1).dropna(how='all',axis=0)
    distance_matrix.set_index(position_matrix['narmad'].values,inplace=True)
    distance_matrix.columns = position_matrix['narmad'].values
    return distance_matrix

def add_lagged_rows(data:pd.DataFrame, armad:str|int,  tol:int = 0, lag:int = 1, order:int = None)->pd.DataFrame:
    """
    Create a dataframe with the lagged samples of any trap. The rows are added only if the time difference between the 
    previous sample and the current one is equal to  14 days + a tolerance. 

    Parameters:
    data: dataframe with the complete data 
    armad: number of trap to be analyzed
    tol: tolerance in days for the time difference between the previous sample and the current one
    lag: number of lagged samples to be added
    order: cardinal number representing the order proximity. 0 is the trap itself, 1 is the closest trap, etc...

    Returns:
    df: dataframe with the lagged samples 

    

    """
    df = row_with_value(data,'narmad', armad)[['nplaca','novos','dtcol']].sort_values('dtcol')
    df['dtcol'] = df['dtcol'].diff()    
    df['good_dates'] = df['dtcol'].apply([lambda x: np.nan if x.days > 14 + tol or x.days < 14 - tol else 1])
    
    for count in range(1,lag+1):
        df[f'novos_lag_{count}_close_{order}'] = df['novos'].shift(count)*df['good_dates'].shift(count-1)
    df.drop(columns=['good_dates'],inplace=True)
    df.set_index('nplaca',inplace=True)
    return df