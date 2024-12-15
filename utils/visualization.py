import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from selenium import webdriver
import plotly.express as px
import itertools
import pdb



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
        offset = 0
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
            y=0,
            font=dict(size=10),

            traceorder='normal'
        ),
        template='plotly_white'  # Optional: use a white background
    )

    # Show the plot
    return fig

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

def convert_html_to_image(html_path):
    # Convert the local path to a file URL
    html_file_url = Path(html_path).resolve().as_uri()

    # Set up Selenium WebDriver (Chrome in this case)
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run Chrome in headless mode
    options.add_argument('--disable-gpu')  # Disable GPU acceleration
    options.add_argument('--window-size=1200x800')  # Set window size for screenshot

    driver = webdriver.Chrome(options=options)

    # Open the HTML file using the file URL
    driver.get(html_file_url)

    # Take a screenshot and save it as an image
    image_path = html_path.replace('.html', '.png')
    driver.save_screenshot(image_path)

    # Close the driver
    driver.quit()

    return image_path


def NovosxRelation_plot_traps(data, lags, ntraps, relation, jitter_strength=1, wrt='lags'):
    """
    Function to plot scatter plots of a relation of lagged traps vs novos for each trap and lag
    Traps are constant for each plot and the relation is calculated for the change of lags

    Parameters:
    data (DataFrame): The input data containing lagged traps and novos
    lags (int): The number of lags to consider
    ntraps (int): The number of traps to consider
    relation (str): The relation to plot ('median' or 'mean' or 'sum')
    jitter_strength (float): The strength of jitter to add to the plot

    Returns:
    None    
    """



    for j in range(ntraps): 
        
        # calculate the relation for the current trap
        lags_list = [f'trap{j}_lag{i}' for i in range(1, lags + 1)] # list of column name for all lagged values for the neighbor trap
        if relation == 'median':
            relation_data = data[lags_list].median(axis=1)
        elif relation == 'mean':
            relation_data = data[lags_list].mean(axis=1)
        elif relation == 'sum':
            relation_data = data[lags_list].sum(axis=1)
        else:
            raise ValueError("relation must be 'median' or 'mean'")
        

        # Add jitter to the data
        data_plot = {}
        jitter_strength = 0.5
        jitter1 = np.random.uniform(-jitter_strength, jitter_strength, size=data.shape[0])
        data_plot['relation'] = relation_data + jitter1
        jitter2 = np.random.uniform(-jitter_strength, jitter_strength, size=data.shape[0])

        data_plot['novos'] = data['novos'] + jitter2

        # Create scatter plot
        fig = px.scatter(data_plot, y='relation', x='novos', title=f'Scatter Plot of {relation} Traps vs. Novos - trap {j} lags{[i for i in range(1, lags + 1)]}')
        fig.update_traces(marker=dict(size=1,color='black'))  # 'size=1' as a close equivalent to 's=0.01'
        fig.update_layout(yaxis_title=f"{relation} of traps", xaxis_title="Novos", xaxis=dict(range=[0, 250]),yaxis=dict(range=[0, 250]))
        fig.show()



def NovosxRelation_plot_lags(data, lags, ntraps, relation, jitter_strength=1, wrt='lags'):
    """
    Function to plot scatter plots of a relation of lagged traps vs novos for each trap and lag
    Lags are constant for each plot and the relation is calculated for the change of neighbors traps

    Parameters:
    data (DataFrame): The input data containing lagged traps and novos
    lags (int): The number of lags to consider
    ntraps (int): The number of traps to consider
    relation (str): The relation to plot ('median' or 'mean'or 'sum')
    jitter_strength (float): The strength of jitter to add to the plot

    Returns:
    None    
    """



    for j in range(1, lags + 1):
        
        # calculate the relation for the current trap
        lags_list = [f'trap{i}_lag{j}' for i in range(ntraps)] # list of column name for all lagged values for the neighbor trap
        if relation == 'median':
            relation_data = data[lags_list].median(axis=1)
        elif relation == 'mean':
            relation_data = data[lags_list].mean(axis=1)
        elif relation == 'sum':
            relation_data = data[lags_list].sum(axis=1)
        else:
            raise ValueError("relation must be 'median' or 'mean'")
        

        # Add jitter to the data
        data_plot = {}
        jitter_strength = 1
        jitter1 = np.random.uniform(-jitter_strength, jitter_strength, size=data.shape[0])
        jitter2 = np.random.uniform(-jitter_strength, jitter_strength, size=data.shape[0])
        data_plot['relation'] = relation_data + jitter1
        data_plot['novos'] = data['novos'] + jitter2

        # Create scatter plot
        fig = px.scatter(data_plot, y='relation', x='novos', title=f'Scatter Plot of {relation} Traps vs. Novos - trap {[i for i in range(ntraps)]} lags{j}')
        fig.update_traces(marker=dict(size=1,color='black'))  # 'size=1' as a close equivalent to 's=0.01'
        fig.update_layout(yaxis_title=f"{relation} of traps", xaxis_title="Novos", xaxis=dict(range=[0, 250]),yaxis=dict(range=[0, 250]))
        fig.show()





def NovosxRelation_plot_all(data, lags, ntraps, relation, jitter_strength=1, wrt='lags', plot_title = None):
    """
    Function to plot scatter plots of a relation of lagged traps vs novos for each trap and lag
    Lags are constant for each plot and the relation is calculated for the change of neighbors traps

    Parameters:
    data (DataFrame): The input data containing lagged traps and novos
    lags (int): The number of lags to consider
    ntraps (int): The number of traps to consider
    relation (str): The relation to plot ('median' or 'mean'or 'sum' or 'naive')
    jitter_strength (float): The strength of jitter to add to the plot

    Returns:
    None    
    """


    
    # calculate the relation for the current trap
    lags_list = [f'trap{i}_lag{j}' for i,j in itertools.product(range(ntraps),range(1,lags+1))] # list of column name for all lags and neighbors traps
    if relation == 'median':
        relation_data = data[lags_list].median(axis=1)
    elif relation == 'mean':
        relation_data = data[lags_list].mean(axis=1)
    elif relation == 'sum':
        relation_data = data[lags_list].sum(axis=1)
    elif relation == 'naive':
        relation_data = data['trap0_lag1']
        
    
    else:
        raise ValueError("relation not defined")
    

    # Add jitter to the data
    data_plot = {}
    jitter_strength1 = 0.4
    jitter_strength2 = 0.4

    jitter1 = np.random.uniform(-jitter_strength1, jitter_strength1, size=data.shape[0])
    jitter2 = np.random.uniform(-jitter_strength2, jitter_strength2, size=data.shape[0])
    data_plot['relation'] = relation_data + jitter1
    data_plot['novos'] = data['novos'] + jitter2

    # Create scatter plot
    if plot_title == None:
        plot_title = f'Scatter Plot of {relation} of lagged Traps vs. Novos'
    fig = px.scatter(data_plot, y='novos', x='relation', title=plot_title )
    fig.update_traces(marker=dict(size=1,color='black'))  # 'size=1' as a close equivalent to 's=0.01'
    fig.update_layout(xaxis_title=f"{relation} of traps", yaxis_title="Novos")
    fig.show()



def plot_results_pytorch(variable_plot, list_plot,version_list,size,epochs,mt): 
    """
    variable_plot = ['total_loss', 'loss_class', 'loss_reg', 'acc_class', 'acc_reg', 'error_reg']
    
    Epochs must be used!
    """
 

    plt.figure(figsize=(15, 6))  
    for i in range(size):   
        y = list(map((lambda x: x/100 if x > 1 else x),list_plot[i][variable_plot]))
        x = range(1, epochs+1)
        plt.plot(y, x, label='Version {}'.format(version_list[i]))
        plt.xlabel('Epoch')
        plt.ylabel(f'{variable_plot}')
        plt.legend()
        plt.title(f'Model {variable_plot} {"TODO"}: {mt}')

    plt.show()




def surface_plot(z, ztitle, plot_title = '3D Surface Plot'): 
    """
    Plot a 3D surface plot using Plotly

    Parameters:
    z: Pivoted DataFrame containing the values to be plotted 
    ztitle : Title of the z axis
    plot_title: Title of the plot
    
    """
    z.index = z.index.astype(int)
    z.columns = z.columns.astype(int)
    z = z.sort_index(ascending=True)
    z = z.sort_index(axis =1,ascending=True)
    z = z.interpolate(method='linear', axis=0)
    fig = go.Figure(data=[go.Surface(z=z.values, x=z.columns, y=z.index)])

    # Update layout for better readability
    fig.update_layout(
        title=plot_title,
        scene=dict(
            xaxis_title='Lags (X)', 
            yaxis_title='Number of neighbors (Y)',
            zaxis_title= ztitle,

        ),
        coloraxis_colorbar=dict(title="Scale"),
        width=1000,  # Increase width
        height=800,   # Increase height

    )

    # Show plot
    fig.show()












