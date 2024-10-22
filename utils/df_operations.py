
import pandas as pd
from typing import Callable, Optional
from IPython.display import display
import numpy as np





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

