#Importing libraries
import sys
import os
import pdb; 
import pandas as pd
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils import df_operations

# TODO: Refactor this code to unite all the data treatment proccess, from the raw PBH data until 
# the final_df to be used to create NN data, including add_feature_to_df.ipynb, dtcol_dtinstal_treatment.ipydb,
# external_data_incorporation and some functions of treate_df_initial_exploration.ipynb and 
# df_initial_visualization.ipynb. Also, change the name of the file to something more meaningfu.

# Importing the dataset and getting informations about it
masterdata = pd.read_stata('../data/MasterDataExtendStata12.dta')
original_shape = masterdata.shape
data = pd.read_csv('../data/final_data.csv',parse_dates=['dtcol']) #dataset with locations
unique_values = df_operations.info_dict_col(masterdata,masterdata.columns,df_operations.get_col_unique_values)
col_types = df_operations.info_dict_col(masterdata, masterdata.columns, df_operations.get_col_types)
active_traps = pd.read_excel('../data/complementar/OVITRAMPAS_ATIVAS.xlsx')




### Remove useless columns 
remove_cols = []
for col,value in unique_values.items():
     sum_dropped = 0 
     if value.shape[0] == 1:
        masterdata.drop(col, axis=1, inplace=True)
        remove_cols.append(col)
        sum_dropped += 1
for col in remove_cols:
    del unique_values[col]
    del col_types[col]


#Substituting NaN values with 0
masterdata[['eclod','desid']] = masterdata[['eclod','desid']].fillna(0)


# NaN means lack of crutial information
nan_eggs = df_operations.print_rows_with_nan(masterdata, 'novos',True,False)
masterdata.drop(nan_eggs.index, axis=0, inplace=True)
nan_rows_coord = df_operations.print_rows_with_nan(masterdata[['narmad','coordx','coordy']], 'coordy',True)


# Treat file with coordinates to merge with masterdata
active_traps['OVITRAMPA'] = active_traps['OVITRAMPA'].apply(lambda x: int(f'90{x}')) # add 90 to the beginning of the trap number

# merge add latitude and longitute to masterdata
new_active_traps = active_traps[['OVITRAMPA','LATITUDE','LONGITUDE','X','Y']].rename(
    columns={'OVITRAMPA':'narmad', 'LATITUDE':'latitude','LONGITUDE':'longitude','X':'coordx2','Y':'coordy2'})

final_df = pd.merge(masterdata, new_active_traps, on='narmad', how='left')
# Save treated dataframe
final_df.to_csv('../data/final_data.csv', sep=',',decimal='.',index=False)