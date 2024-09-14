import pandas as pd
from IPython.display import display
import xml.etree.ElementTree as ET

def print_rows_with_nan(df:pd.DataFrame,col:str)->None:
    """
    Print rows with NaN values in the column col of the dataframe df.

    Parameters:
    df: pandas dataframe
    col: string

    Returns:
    None
    """

    display(df[df[col].isna()])
    

def get_col_types(df:pd.DataFrame,col:str)->pd.DataFrame:
    """
    Get the data types of the columns in the dataframe df.

    Parameters:
    df: pandas dataframe

    Returns:
    pandas dataframe
    """

    return df[col].apply(type).unique()


def get_col_unique_values(df:pd.DataFrame,col:str)->pd.DataFrame:
    """
    Get the unique values of the columns in the dataframe df.

    Parameters:
    df: pandas dataframe

    Returns:
    pandas dataframe
    """

    return df[col].unique()



