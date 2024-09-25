import pandas as pd
from IPython.display import display
import numpy as np
from typing import Callable, Optional
import matplotlib.pyplot as plt
import pygame


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

def df_difference(df1,df2):
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


def estimate_transform(source_points, target_points):
    # Calculate centroids
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # Center the points
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    # Compute the covariance matrix
    H = np.dot(source_centered.T, target_centered)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute translation
    translation = centroid_target - np.dot(R, centroid_source)

    return R, translation


def hist_html(df,trap,create_hist=True):
    if create_hist:
        # Generate a histogram
        plt.figure(figsize=(6, 4))
        plt.hist(df, bins=1000, color='black', edgecolor='black')
        plt.title(f'Histograma de contagem de ovos - armadilha {trap}')
        plt.xlabel('Contagem de ovos')
        plt.ylabel('Frequência')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f'./histograms/histogram_{trap}.png', bbox_inches='tight', dpi=300)
        plt.close()  # Close the plot to avoid displaying it in the output

    return f'./histograms/histogram_{trap}.png'


def play_finish_song(addr = 'D:/HD_backup/Pedro/Músicas/Músicas/Sinfonia To Cantata # 29.mp3'):
    # Initialize the mixer
    pygame.mixer.init()
    # Load the MP3 file
    pygame.mixer.music.load(addr)
    # Play the MP3 file
    pygame.mixer.music.play()

def stop_finish_song():
    pygame.mixer.music.stop()
    # Optional: Clean up the mixer
    pygame.mixer.quit()