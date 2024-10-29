
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def plot_time_series_yearly(df:pd.DataFrame, column:str, xaxis:str ,title:str, xaxis_title:str, yaxis_title:str, global_plot:bool = False)->None:
    """
    Plot time series from column in a year by year basis using plotly

    Parameters:
    df: dataframe containing the data
    column: column of the variable to plot
    title: title of the plot
    xaxis_title: title of the x axis
    yaxis_title: title of the y axis
    xaxis: str to determine if the x axis is in week or month
    global_plot: bool to determine if the plot will be separated in years or not

    Returns:
    None
    
    
    """
    if xaxis == 'week':
        xaxis = 'semepi'
        offset = 100
    elif xaxis == 'month':
        xaxis = 'mesepid'
        offset = 0
    else:
        raise ValueError('xaxis must be either "week" or "month"')
    

    if global_plot:
        global_title = ' (Global)'
        if xaxis == 'semepi':
            global_offset = 52
        elif xaxis == 'mesepid':
            global_offset = 12
    else:
        global_title = ''
        global_offset = 0



    # Create a new figure
    fig = go.Figure()
    j = 0
    # Loop through unique years in the 'anoepid' column
    for year in df['anoepid'].unique():
        df_year = df[df['anoepid'] == year]
        # Add a line for each year
        fig.add_trace(go.Scatter(
            x=df_year[xaxis] - offset + j * global_offset,
            y=df_year[column],
            mode='lines',
            name=str(year)  # Convert year to string for the legend
        ))
        j += 1

    # Update layout
    fig.update_layout(
        title= title + global_title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title='Ano epidemiológico',
        legend=dict(
            title=dict(font=dict(size=12)),
            x=1,  # Position legend outside of the plot area
            y=1,
            traceorder='normal'
        ),
        template='plotly_white'  # Optional: use a white background
    )

    # Show the plot
    fig.show()

def plot_time_series_cat(df:pd.DataFrame, column:str, xaxis:str ,title:str, xaxis_title:str, yaxis_title:str,):
    """
    Subplot time series from column in a year by year basis using plotly dividingby category

    Parameters:
    df: dataframe containing the data
    column: column of the variable to plot
    title: title of the plot
    xaxis_title: title of the x axis
    yaxis_title: title of the y axis
    xaxis: str to determine if the x axis is in week or month

    Returns:
    None
    
    
    """

    if xaxis == 'week':
        xaxis = 'semepi'
        offset = 100
    elif xaxis == 'month':
        xaxis = 'mesepid'
        offset = 0
    else:
        raise ValueError('xaxis must be either "week" or "month"')
    




    # Get unique categories and years
    categories = ['A2','A1','M ','B ']
    years = df['anoepid'].unique()

    # Create a 2x2 subplot figure
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f'Média de ovos para {cat}' for cat in categories])

    # Loop through each category and year, adding each plot to the correct subplot position
    row, col = 1, 1
    for i, category in enumerate(categories):
        for year in years:
            data_year = df[df['anoepid'] == year]
            data_year_cat = data_year[data_year['GerCat'] == category]
            fig.add_trace(go.Scatter(
                x=data_year_cat[xaxis] - offset, 
                y=data_year_cat[column],
                mode='lines+markers',
                name=str(year),
                showlegend=(i == 0)  # Show legend only in the first subplot
            ), row=row, col=col)

        # Move to the next subplot position
        col += 1
        if col > 2:  # Reset column and move to next row
            col = 1
            row += 1

    # Update layout with titles and axis labels
    fig.update_layout(
        title= title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=800,
        showlegend=True  # Show the legend once for clarity
    )

    # Show the figure
    fig.show()



def time_series_html(df:pd.DataFrame,trap:str) -> str:
    """
    Generate a Pareto plot of the data of a specific trap so it can be saved as an image file and used in the html

    Parameters:
    df: pandas DataFrame with the novos and dtcol
    trap: string refering to the trap number

    Returns:
    str: path to the image file
    """

    df.sort_values('dtcol', inplace=True)  # Sort the values by date
    

    # Generate a histogram
    plt.figure(figsize=(6, 4))
    
    
    
    
    
    plt.plot(df['dtcol'] ,df['novos'], color='c')
    plt.title(f'Série temporal da contagem de ovos - armadilha {trap}')
    plt.xlabel('Data')
    plt.ylabel('Contagem de ovos')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'../results/time_series/time_series_{trap}.png', bbox_inches='tight', dpi=300)
    plt.close()  # Close the plot to avoid displaying it in the output
    return f'../results/time_series/time_series_{trap}.png'