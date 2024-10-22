import pandas as pd
import numpy as np





def create_lagged_rows(data:pd.DataFrame, trap:str|int,  tol:int = 3, lag:int = 1)->pd.DataFrame:
    """
    Create a dataframe with the lagged samples of any trap. The rows are added only if the time difference between the 
    previous sample and the current one is equal to  14 days + a tolerance. 

    Parameters:
    data: dataframe with the complete data 
    trap: number of the other trap to be analyzed
    tol: tolerance in days for the time difference between the previous sample and the current one
    lag: number of lagged samples to be added

    Returns:
    df: dataframe with the lagged samples 

    """
    df = row_with_value(data,'narmad', trap)[['nplaca','narmad','novos','dtcol']].sort_values('dtcol')
    df['dtcol_diff_1'] = df['dtcol'].diff().dt.days    
    df['good_dates'] = df['dtcol_diff_1'].apply([lambda x: np.nan if x > 14 + tol or x < 14 - tol else 1])
    
    for count in range(1,lag):
        df[f'dtcol_diff_{count+1}'] = df[f'dtcol_diff_{count}'] + df['dtcol_diff_1'].shift(count) 

    for count in range(1,lag+1):
        df[f'novos_lag_{count}'] = df['novos'].shift(count)*df['good_dates'].shift(count-1)
   
    df.set_index('nplaca',inplace=True)
    return df

'''

def merge_series(df1:pd.DataFrame, df2:pd.DataFrame, on:str = 'dtcol')->pd.DataFrame:
    """
    Merge two dataframes representing time series data. The dataframes are merged on the column "on".

    Parameters:
    df1: original df that will be filled with lagged data
    df2: dataframe containing lagged data of some traps
    on: string representing the column to be used as key for the merge. Initially, it is the collection date.
    Traps are associated according to the most recent collection before the original trap.


    Returns:
    pandas dataframe with the merged data
    """
    for original_trap in df1['narmad'].iterrows():
        diff_date = original_trap['dtcol'] - df2['dtcol']
        other_trap_nplaca = diff_date[diff_date>=0].min()['nplaca'] #get the most recent collection date before the original trap
        other_trap = row_with_value(df2,'nplaca',other_trap_nplaca)
        other_trap[]
        other_trap = other_trap.drop(columns=['nplaca','dtcol'])
        
        
        df1.loc[original_trap] = df1.loc[original_trap].combine_first()
    
    return 
'''