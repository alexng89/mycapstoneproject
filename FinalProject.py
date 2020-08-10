import pandas as pd  
import numpy as np
from pandas_datareader import data, wb
from datetime import datetime, timedelta
import json
import dash
import dash_table
import dash_core_components as dcc 
import dash_html_components as html 
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go



###############################################
############# PRE-DEFINED INPUTS #############
###############################################

# list of tickers for user to choose
with open(r"tickerdata.json", "r") as read_file:
    ticker_data=json.load(read_file)

# state start/end time (always using last 1Y's data; 365 days)
end = datetime.today()
endstr = end.strftime('%d-%b-%y')
start = end - timedelta(days=365)

# specify how many portfolio simulations to run (always 10,000)
num_portfolios = 10000

# define risk-free rate (assume 0 for now)
rf = 0.0


###############################################
################## DASHBOARD ##################
###############################################

app = dash.Dash(__name__)

app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Div(id='intermediate-value', style={'display': 'none'}),
])

home_page = html.Div([
    html.H1('Welcome!'), 
    html.P("Let's get started with your portfolio by selecting from the following list:"),
    html.P(dcc.Dropdown(id='user_inputs', options=ticker_data, multi=True)),
    html.Button('Analyze My Portfolio', id='user_button'),
    html.Br(),
    html.Br(),
    html.I('Disclaimer: This analysis is intended to be used for informational purposes only. It is very important to do your own analysis before making any investment decisions. You should take independent financial advice from a professional in connection with, or independently research and verify, any information that you find from this analysis.'),
    # html.A(dcc.Link('Reset portfolio', href='/')),
    # html.A(html.Button('Reset portfolio'), href='/'),
    html.Div(id='homepage-content'),
])

@app.callback([Output('intermediate-value', 'children'), Output('url', 'pathname')],
              [Input('user_button', 'n_clicks')], 
              [State('user_inputs', 'value')])

def run_analysis(n_clicks, user_inputs):
    if n_clicks is None:   
        raise PreventUpdate
    else:     
        ###############################################
        ####### TASK #1: UNDERSTANDING THE USER #######
        ###############################################

        # create df from user inputs, for use to download yahoo finance data   
        df = pd.DataFrame(user_inputs)

        # get nuber of user securities
        num_securities = len(user_inputs)
        
        # initialize empty list, and then get equal weights via loop
        # change my_weights list into a numpy array for processing later
        my_weights = []
        for i in range(num_securities):
            my_weights.append(1/num_securities)
        my_weights = np.array(my_weights)


        ###############################################
        ########## TASK #2: DATA PROCUREMENT ##########
        ###############################################
        ############ TASK #3: DATA STORAGE ############
        ###############################################

        # download yahoo data and store as dataframe in memory
        df = pd.DataFrame([data.DataReader(ticker, 'yahoo', start, end)['Adj Close'] for ticker in user_inputs]).T
        
        # make df columns as tickers from user inputs
        df.columns = user_inputs


        ###############################################
        ######### TASK #4: PORTFOLIO ANALYSIS #########
        ###############################################
        ##### DEFINE FUNCTIONS FOR DATA ANALYSIS ######
        ###############################################

        # define function to calculate portfolio returns, volatility, and sharpe ratio
        def calc_portfolio_perf(weights, mean_returns, cov, rf):
            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
            sharpe_ratio = (portfolio_return - rf) / portfolio_std
            return portfolio_return, portfolio_std, sharpe_ratio

        # define function to run simulation
        def simulate_random_portfolios(num_portfolios, mean_returns, cov, rf):
            results_matrix = np.zeros((len(mean_returns)+3, num_portfolios))
            for i in range(num_portfolios):
                weights = np.random.random(len(mean_returns))
                weights /= np.sum(weights)
                portfolio_return, portfolio_std, sharpe_ratio = calc_portfolio_perf(weights, mean_returns, cov, rf)
                results_matrix[0, i] = portfolio_return
                results_matrix[1, i] = portfolio_std
                results_matrix[2, i] = sharpe_ratio
                # iterate through the weight vector and add data to results array
                for j in range(len(weights)):
                    results_matrix[j+3, i] = weights[j]            
            results_df = pd.DataFrame(results_matrix.T, columns=['Returns','StDev','Sharpe'] + [ticker for ticker in user_inputs])        
            return results_df


        ###############################################
        ######### TASK #4: PORTFOLIO ANALYSIS #########
        ###############################################
        ####### INTERMEDIATE DATA FOR LATER USE #######
        ###############################################

        # calculate mean returns of each security
        mean_returns = df.pct_change().mean()
        
        # calculate covariance of each security
        cov = df.pct_change().cov()


        ###############################################
        ######### TASK #4: PORTFOLIO ANALYSIS #########
        ###############################################
        ####### RUN SIMULATIONS USING FUNCTIONS #######
        ###############################################

        # get the simulation results
        results_frame = simulate_random_portfolios(num_portfolios, mean_returns, cov, rf)


        ###############################################
        ######### TASK #4: PORTFOLIO ANALYSIS #########
        ###############################################
        ###### GET PERFORMANCE OF USER PORTFOLIO ######
        ###############################################

        # calculate user portfolio performance
        my_port = calc_portfolio_perf(my_weights, mean_returns, cov, rf)
        
        # create df of user portfolio and equal weights, merge them together
        my_df = pd.DataFrame(my_port)
        my_weights = pd.DataFrame(my_weights)
        my_df = my_df.append(my_weights, ignore_index=True).T

        # rename columns to match result_frame from simulations done previously
        my_df.columns = ['Returns','StDev','Sharpe'] + [ticker for ticker in user_inputs]
        
        # merge df of user portfolio and weights into the simulation results frame
        results_frame = results_frame.append(my_df, ignore_index=True)


        ###############################################
        ######### TASK #4: PORTFOLIO ANALYSIS #########
        ###############################################
        ######### GET USER PORTFOLIO SNAPSHOT #########
        ###############################################

        # create a list to get 1Y, 6M, 3M, 11M, 1D price changes
        px_change_list = [-252, -126, -63, -21, -2]

        # initialize an empty df
        df_FinalPxChg = pd.DataFrame()

        # calculate the % price change for each ticker over each period
        # concatenate results into the empty df
        for i in px_change_list:
            df_PxChg = pd.DataFrame(df.iloc[[i, -1]])
            df_PxChg = df_PxChg.pct_change().reset_index().iloc[:, 1:].tail(1).T
            df_FinalPxChg = pd.concat([df_FinalPxChg, df_PxChg], axis=1)

        # reset the % price change df for merging
        df_FinalPxChg = df_FinalPxChg.reset_index()

        # create df from last price of raw data, drop the date column
        # transpose the data and reset index
        df_LastPrice = pd.DataFrame(df.tail(1)).T
        df_LastPrice = df_LastPrice.reset_index()

        # merge the % price change df into snapshot
        df_snapshot = pd.merge(df_LastPrice, df_FinalPxChg, on='index')

        # rename columns and then reorder them
        df_snapshot.columns = ['Ticker', 'Last Price', '1Y Ret', '6M Ret', '3M Ret', '1M Ret', '1D Ret']
        df_snapshot = df_snapshot[['Ticker', 'Last Price' , '1D Ret', '1M Ret', '3M Ret', '6M Ret', '1Y Ret']]


        ###############################################
        ######### TASK #4: PORTFOLIO ANALYSIS #########
        ###############################################
        ####### DATA CLEANING FOR VISUALIZATION #######
        ###############################################

        # locate portfolio with highest Sharpe Ratio
        max_sharpe_port = results_frame.iloc[results_frame['Sharpe'].idxmax()]

        # locate portfolio with minimum standard deviation
        min_vol_port = results_frame.iloc[results_frame['StDev'].idxmin()]

        # locate user's portfolio
        user_port = results_frame.iloc[results_frame.tail(1).index.item()]

        # create summary dataframe of the above three portfolios
        result = pd.concat([max_sharpe_port.to_frame(), min_vol_port.to_frame(), user_port.to_frame()], axis=1, sort=False)
        result = result.reset_index()
        result.columns = ['Ticker', 'Optimal Portfolio', 'Min Vol Portfolio', 'User Portfolio']

        # create result_summary for just returns, stdev and sharpe
        result_summary = result.set_index('Ticker').T
        result_summary['Returns'] = result_summary['Returns'].map('{:,.2%}'.format)
        result_summary['StDev'] = result_summary['StDev'].map('{:,.2%}'.format)
        result_summary['Sharpe'] = result_summary['Sharpe'].map('{:,.2f}'.format)
        result_summary = result_summary.iloc[:,:3]
        result_summary.insert(loc=0, column='', value=['Optimal Portfolio', 'Min Vol Portfolio', 'User Portfolio'])

        # create a JSON file to hold data of all processed data and store into a hidden div
        datasets = {'df_1': results_frame.to_json(orient='split', date_format='iso'), 
                    'df_2': result_summary.to_json(orient='split', date_format='iso'),
                    'df_3': max_sharpe_port.to_json(orient='split', date_format='iso'),
                    'df_4': min_vol_port.to_json(orient='split', date_format='iso'),
                    'df_5': user_port.to_json(orient='split', date_format='iso'),
                    'df_6': result.to_json(orient='split', date_format='iso'),
                    'df_7': df_snapshot.to_json(orient='split', date_format='iso'),
                    }

        return json.dumps(datasets), '/snapshot'


###############################################
###### TASK #5: PORTFOLIO VISUALIZATION #######
###############################################
############# PORTFOLIO SNAPSHOT ##############
###############################################

snapshot_layout = html.Div([
    html.I(f'Source for all charts and tables: Yahoo Finance as of {endstr}. Past performance is not indicative of future results.'),    
    html.H1('Portfolio Snapshot'),    
    html.Div(id='table1-content'),
    html.H1('Simulation Results'),    
    html.I(f'10,000 portfolio simulations ran using average returns from last 1Y.'),    
    html.Div(id='table2-content'), 
    dcc.Graph(id='myGraph'),
    html.H1('Portfolio Weights'),    
    html.Div(id='table3-content'),
    html.Br(),
    html.Br(),
    dcc.Link('Reset portfolio', href='/'),
])

@app.callback([Output('myGraph', 'figure'), Output('table1-content', 'children'), Output('table2-content', 'children'), Output('table3-content', 'children')],
             [Input('intermediate-value', 'children')])

def snapshot_table(clean_data):
    
    # load the processed data
    datasets = json.loads(clean_data)

    # load the dataframe used for plotting the main scatter plot
    results_frame = pd.read_json(datasets['df_1'], orient='split')
    
    # load the stylized table consisting the returns/stdev/sharpe
    result_summary = pd.read_json(datasets['df_2'], orient='split')

    # load the series containing the max sharpe portfolio
    max_sharpe_port = pd.read_json(datasets['df_3'], orient='split', typ='series')

    # load the series containing the minimum volatility portfolio
    min_vol_port = pd.read_json(datasets['df_4'], orient='split', typ='series')

    # load the series containing the user portfolio (equal-weighted)
    user_port = pd.read_json(datasets['df_5'], orient='split', typ='series')

    # load table containing weights of securities in each portfolio
    # style it here
    result = pd.read_json(datasets['df_6'], orient='split')
    result = result.iloc[3:]
    result['Optimal Portfolio'] = result['Optimal Portfolio'].map('{:,.2%}'.format)
    result['Min Vol Portfolio'] = result['Min Vol Portfolio'].map('{:,.2%}'.format)
    result['User Portfolio'] = result['User Portfolio'].map('{:,.2%}'.format)

    # load table containing portfolio snapshot
    # style it here
    df_snapshot = pd.read_json(datasets['df_7'], orient='split')
    df_snapshot['Last Price'] = df_snapshot['Last Price'].map('{:,.2f}'.format)
    df_snapshot['1D Ret'] = df_snapshot['1D Ret'].map('{:,.2%}'.format)
    df_snapshot['1M Ret'] = df_snapshot['1M Ret'].map('{:,.2%}'.format)
    df_snapshot['3M Ret'] = df_snapshot['3M Ret'].map('{:,.2%}'.format)
    df_snapshot['6M Ret'] = df_snapshot['6M Ret'].map('{:,.2%}'.format)
    df_snapshot['1Y Ret'] = df_snapshot['1Y Ret'].map('{:,.2%}'.format)

    # create scatterplot of simulated and the above three portfolios (max sharpe/min vol/user portfolio)
    # sharpe ratio used for color scale bar
    fig = go.Figure(data=[
                    go.Scatter(x=results_frame.StDev, y=results_frame.Returns, customdata=results_frame.Sharpe, hovertemplate = '<b>Ret:</b> %{y:,.2%}' + '<br><b>Vol:</b> %{x:,.2%}' + '<br><b>Sharpe:</b> %{customdata:,.2f}', marker=dict(color=results_frame.Sharpe,colorbar=dict(title='Sharpe Ratio<br>(Higher is better)'), colorscale='blues'), mode='markers', name='Simulated Portfolios'), 
                    go.Scatter(x=pd.Series(max_sharpe_port[1]), y=pd.Series(max_sharpe_port[0]), customdata=pd.Series(max_sharpe_port[2]), hovertemplate = '<b>Ret:</b> %{y:,.2%}' + '<br><b>Vol:</b> %{x:,.2%}' + '<br><b>Sharpe:</b> %{customdata:,.2f}', mode='markers', marker=dict(symbol='star', size=20, color='gold'), name='Max Sharpe Portfolio'),
                    go.Scatter(x=pd.Series(min_vol_port[1]), y=pd.Series(min_vol_port[0]), customdata=pd.Series(min_vol_port[2]), hovertemplate = '<b>Ret:</b> %{y:,.2%}' + '<br><b>Vol:</b> %{x:,.2%}' + '<br><b>Sharpe:</b> %{customdata:,.2f}', mode='markers', marker=dict(symbol='star', size=20, color='green'), name='Minimum Volatility Portfolio'),
                    go.Scatter(x=pd.Series(user_port[1]), y=pd.Series(user_port[0]), customdata=pd.Series(user_port[2]), hovertemplate = '<b>Ret:</b> %{y:,.2%}' + '<br><b>Vol:</b> %{x:,.2%}' + '<br><b>Sharpe:</b> %{customdata:,.2f}', mode='markers', marker=dict(symbol='star', size=20, color='red'), name='User Portfolio'),
                    ])

    # style figure layout
    fig.update_layout(xaxis_title='Portfolio Volatility', 
                      yaxis_title='Portfolio Returns', 
                      xaxis=dict(tickformat=',.1%'), 
                      yaxis=dict(tickformat=',.1%'), 
                      legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', yanchor='bottom', y=1, xanchor='right', x=1)
                      )
        
    # table1 containing portfolio snapshot
    table1 = dash_table.DataTable(data=df_snapshot.to_dict('records'), 
                                  columns=[{'id': c, 'name': c} for c in df_snapshot.columns], 
                                  style_cell={'textAlign': 'center'}, 
                                  style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}], 
                                  style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                                  )

    # table2 containing portfolio stats
    table2 = dash_table.DataTable(data=result_summary.to_dict('records'), 
                                  columns=[{'id': c, 'name': c} for c in result_summary.columns], 
                                  style_cell={'textAlign': 'center'}, 
                                  style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}], 
                                  style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                                  )

    # table3 containing portfolio weights
    table3 = dash_table.DataTable(data=result.to_dict('records'), 
                                  columns=[{'id': c, 'name': c} for c in result.columns], 
                                  style_cell={'textAlign': 'center'}, 
                                  style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}], 
                                  style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                                  )

    return fig, table1, table2, table3


# Update the home page
@app.callback(Output('page-content', 'children'),
             [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/snapshot':
        return snapshot_layout
    elif pathname == '/':
        return home_page
    else:
        return "Page does not exist"


if __name__ == '__main__':
    app.run_server(debug=True, port=4000)



