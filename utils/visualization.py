
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


def time_series_html(df:pd.DataFrame,trap:str) -> str:
    """
    Generate a Pareto plot of the data of a specific trap so it can be saved as an image file and used in the html

    Parameters:
    df: pandas DataFrame with the number of traps, novos, dtcol
    trap: string refering to the trap number

    Returns:
    str: path to the image file
    """
    # Generate a histogram
    plt.figure(figsize=(6, 4))
    
    
    
    
    
    plt.plot(df, bins=1000, color='black', edgecolor='black')





    plt.title(f'Série temporal da contagem de ovos - armadilha {trap}')
    plt.xlabel('Amostra')
    plt.ylabel('Contagem de ovos')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'./results/time_serie/time_series_{trap}.png', bbox_inches='tight', dpi=300)
    plt.close()  # Close the plot to avoid displaying it in the output

    return f'./time_serie/time_series_{trap}.png'