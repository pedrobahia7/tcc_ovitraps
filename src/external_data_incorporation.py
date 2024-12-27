import pandas as pd
import numpy as np 
import os 
from geopy.distance import geodesic


info_df = pd.read_csv('../data/final_data.csv')
info_df.dropna(subset=['latitude', 'longitude'], inplace=True)  

# Load dfs 
root = "../data/external_data/dfs_final/"
df_dict = {}
for file in os.listdir(root):
    if file.endswith(".csv"):
        df = pd.read_csv(root + file, index_col=0)
        df_dict[file] = df
        
# Agregate daily data by mean
df_dict_agg = {}
for key in df_dict.keys():
    df_dict[key]['date'] = df_dict[key].index.map(lambda x: pd.Timestamp(x.split(" ")[0]))
    df = df_dict[key].groupby('date').mean()
    df.index = df.index.map(lambda x: pd.Timestamp(x))
    df = df[(df.index >= info_df['dtinstal'].min()) 
                        & (df.index <= info_df['dtcol'].max())]
    
    # Fill missing dates
    date_range = pd.date_range(pd.Timestamp(info_df['dtinstal'].min()), df.index.min() - pd.Timedelta(days=1) , freq='D')
    nan_rows_min = pd.DataFrame(np.nan, index=date_range, columns=df.columns)

    date_range = pd.date_range( df.index.max() + pd.Timedelta(days=1), info_df['dtcol'].max() , freq='D')
    nan_rows_max = pd.DataFrame(np.nan, index= date_range, columns=df.columns)
    df = pd.concat([nan_rows_min, df, nan_rows_max])
    df['latitude'].interpolate(method='ffill', inplace=True)
    df['longitude'].interpolate(method='ffill', inplace=True)
    df['latitude'].interpolate(method='backfill', inplace=True)
    df['longitude'].interpolate(method='backfill', inplace=True)
    
    df_dict_agg[key] = df

# Create Date range, Distance, Temperature, Precipitation and Umidity arrays
distance_array = np.zeros((2,len(df_dict_agg)))
i = 0
temp_df = pd.DataFrame()
prec_df = pd.DataFrame()
umid_df = pd.DataFrame()

for key in df_dict_agg.keys():
    distance_array[0, i] = df_dict_agg[key]['latitude'].unique()[0]  
    distance_array[1, i] = df_dict_agg[key]['longitude'].unique()[0]

    temp_df = pd.concat([temp_df, df_dict_agg[key]['Temperatura']],axis=1)
    prec_df = pd.concat([prec_df, df_dict_agg[key]['Precipitacao']],axis=1)
    umid_df = pd.concat([umid_df, df_dict_agg[key]['Umidade']],axis=1)
    i += 1

temp_array = temp_df.values
prec_array = prec_df.values
umid_array = umid_df.values

date_range_array = np.array(df_dict_agg[key].index)



final_dict = {'nplaca':[], 'Temperatura_1week_bfr':[], 'Precipitacao_1week_bfr':[], 'Umidade_1week_bfr':[]}

# Begin Loop
for narmad in info_df['narmad'].unique():
    info_df[info_df['narmad'] == narmad]
        
    armad_df = info_df[info_df['narmad'] == narmad]
   
    armad_lat =  armad_df['latitude'].unique()[0]
    armad_lon =  armad_df['longitude'].unique()[0]

    
    # Calculate distance between armad and each weather station
    distance = np.zeros(distance_array.shape[1])
    for i in range(distance_array.shape[1]):
        distance[i] = geodesic((armad_lat, armad_lon), (distance_array[0,i], distance_array[1,i])).km
    # nplaca loop
    for placa in armad_df['nplaca'].unique():
        
        # Get the variables for the specified date range
        index_end = np.where(date_range_array == np.datetime64(info_df[info_df['nplaca'] == placa].iloc[0]['dtinstal']))[0][0]
        index_begin = index_end - 7
        armad_umid = umid_array[index_begin:index_end]
        armad_temp = temp_array[index_begin:index_end]
        armad_prec = prec_array[index_begin:index_end]

        # Calculate weights to each weather station. The weights are the inverse of the distance between the armad and the weather station
        weights = np.tile(distance, (armad_umid.shape[0], 1))
        mask = ~np.isnan(armad_umid)
        weights = (weights* mask)
        weights = np.nan_to_num((1/weights), nan=0, posinf=0, neginf=0)
        weights = weights/weights.sum(axis=1,keepdims=True)

        # Calculate the final values of each variable for the armad
        final_temp = np.nansum(armad_temp*weights, axis=1).mean()
        final_prec = np.nansum(armad_prec*weights, axis=1).mean()
        final_umid = np.nansum(armad_umid*weights, axis=1).mean()

        # save to dict
        final_dict['nplaca'].append(placa)
        final_dict['Temperatura_1week_bfr'].append(final_temp)
        final_dict['Precipitacao_1week_bfr'].append(final_prec)
        final_dict['Umidade_1week_bfr'].append(final_umid)

final_df = pd.DataFrame(final_dict)
info_df = info_df.merge(final_df, on='nplaca', how='left')

info_df.to_csv('../data/final_data.csv', index=False)

