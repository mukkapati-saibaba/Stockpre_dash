#!/usr/bin/env python
# coding: utf-8

# In[1]:


import base64
import io
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX



# In[3]:


from jupyter_dash import JupyterDash

# Create the app
app = JupyterDash(__name__)


# In[4]:


# Define the layout
app.layout = html.Div([
    html.Div([
        html.Img(src='assets/logo.png', className='logo'),
        html.H1("Stock Price Prediction Dashboard", className='dashboard-title')
    ], className='header'),

    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.Button('Upload File', id='upload-button', n_clicks=0)
            ]),
            style={'textAlign': 'center', 'marginTop': '20px'},
            multiple=False
        ),
    ], className='upload-container'),

    html.Div([
        html.Label('Select Date Range:', className='date-picker'),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=None,
            end_date=None,
            display_format='YYYY-MM-DD',
            className='date-picker-input'
        )
    ], className='date-picker'),

    dcc.Loading(
        id="loading-icon",
        type="default",
        children=[
            html.Div(className='graph-container', children=[
                dcc.Graph(id='price-plot'),
                html.Div(className='centered-container', children=[
                    html.Div([
                        html.H3("SARIMAX Results DataFrame", className='table-title'),
                        dash_table.DataTable(
                            id='results-table-sarimax',
                            style_table={'overflowX': 'auto'},
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                            style_cell={'textAlign': 'left'}
                        )
                    ])
                ]),
                dcc.Graph(id='xgboost-plot'),
                html.Div(className='centered-container', children=[
                    html.Div([
                        html.H3("XGBoost Results DataFrame", className='table-title'),
                        dash_table.DataTable(
                            id='results-table-xgboost',
                            style_table={'overflowX': 'auto'},
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                            style_cell={'textAlign': 'left'}
                        )
                    ])
                ]),
                html.Div([
                    html.H3("Model Evaluation Metrics", className='table-title'),
                    html.Div(id='metrics-table', className='metrics-container'),
                    html.Div(id='metrics-interpretation')
                ], className='metrics-container'),
                
                dcc.Graph(id='macd-plot', className='macd-plot'),
                dcc.Graph(id='rsi-plot', className='rsi-plot')
            ])
        ]
    )
])

# Define the function to run the models
def run_model(df):
    # Ensure data is sorted by date
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    
    # SARIMAX Model
    endog = df['Close']
    exog = df[['Close_ma_20', 'Close_ma_50', 'MACD']]
    train_size = int(len(df) * 0.8)
    endog_train = endog[:train_size]
    exog_train = exog[:train_size]
    endog_test = endog[train_size:]
    exog_test = exog[train_size:]

    model1 = SARIMAX(endog_train, exog=exog_train, order=(1, 1, 1),
                     seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False, enforce_invertibility=False)
    results = model1.fit(disp=False)
    forecast_model1 = results.predict(start=len(endog_train), end=len(endog_train) + len(endog_test) - 1, exog=exog_test)

    results_df_sarimax = pd.DataFrame({'Date': df['Date'][train_size:], 'Actual': endog_test, 'Forecast': forecast_model1})
    
    # XGBoost Model
    X = df.drop(columns=['Next_Day', 'Aim', 'Date'])  # Drop non-numeric columns
    y = df['Next_Day']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    results_df_xgboost = pd.DataFrame({'Date': df['Date'][train_size:], 'Actual': y_test, 'Predicted': y_pred})
    
    # Calculate metrics
    mse_model1 = mean_squared_error(endog_test, forecast_model1)
    mae_model1 = mean_absolute_error(endog_test, forecast_model1)
    mape_model1 = np.mean(np.abs((endog_test - forecast_model1) / endog_test)) * 100
    
    xgboost_mse = mean_squared_error(y_test, y_pred)
    xgboost_mae = mean_absolute_error(y_test, y_pred)
    xgboost_r2 = r2_score(y_test, y_pred)
    
    return (df, results_df_sarimax, results_df_xgboost, mse_model1, mae_model1, mape_model1, xgboost_mse, xgboost_mae, xgboost_r2)

# Define the function to parse the uploaded data
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return {}, [], [], 0, 0, 0, 0, 0, 0

        df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime

        df, results_df_sarimax, results_df_xgboost, mse_model1, mae_model1, mape_model1, xgboost_mse, xgboost_mae, xgboost_r2 = run_model(df)

        return df, results_df_sarimax, results_df_xgboost, mse_model1, mae_model1, mape_model1, xgboost_mse, xgboost_mae, xgboost_r2

    except Exception as e:
        print(f"Error processing file: {e}")
        return {}, [], [], 0, 0, 0, 0, 0, 0

# Define the callback to update the output
@app.callback(
    [
        Output('price-plot', 'figure'),
        Output('results-table-sarimax', 'data'),
        Output('results-table-sarimax', 'columns'),
        Output('xgboost-plot', 'figure'),
        Output('results-table-xgboost', 'data'),
        Output('results-table-xgboost', 'columns'),
        Output('metrics-table', 'children'),
        Output('macd-plot', 'figure'),
        Output('rsi-plot', 'figure')
    ],
    [Input('upload-data', 'contents'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')],
    [State('upload-data', 'filename')]
)
def update_output(contents, start_date, end_date, filename):
    if contents is not None:
        df, results_df_sarimax, results_df_xgboost, mse_model1, mae_model1, mape_model1, xgboost_mse, xgboost_mae, xgboost_r2 = parse_contents(contents, filename)

        # Filter data based on the selected date range
        if start_date and end_date:
            mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
            df = df[mask]
            results_df_sarimax = results_df_sarimax[results_df_sarimax['Date'].between(start_date, end_date)]
            results_df_xgboost = results_df_xgboost[results_df_xgboost['Date'].between(start_date, end_date)]
        
        if not df.empty:
            # Recalculate metrics and figures based on filtered data
            price_plot_figure = {
                'data': [
                    go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual Price'),
                    go.Scatter(x=results_df_sarimax['Date'], y=results_df_sarimax['Forecast'], mode='lines', name='Forecasted Price', line={'dash': 'dash'})
                ],
                'layout': go.Layout(
                    title='SARIMAX: Actual vs Forecasted Prices',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price'}
                )
            }

            xgboost_plot_figure = {
                'data': [
                    go.Scatter(x=results_df_xgboost['Date'], y=results_df_xgboost['Actual'], mode='lines', name='Actual Price'),
                    go.Scatter(x=results_df_xgboost['Date'], y=results_df_xgboost['Predicted'], mode='lines', name='Predicted Price', line={'dash': 'dash'})
                ],
                'layout': go.Layout(
                    title='XGBoost: Actual vs Predicted Prices',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price'}
                )
            }
            
            # Create metrics table
            metrics_table = html.Table([
                html.Tr([html.Th("Metric"), html.Th("Value")]),
                html.Tr([html.Td("Mean Squared Error (MSE) - SARIMAX"), html.Td(f"{mse_model1:.2f}")]),
                html.Tr([html.Td("Mean Absolute Error (MAE) - SARIMAX"), html.Td(f"{mae_model1:.2f}")]),
                html.Tr([html.Td("Mean Absolute Percentage Error (MAPE) - SARIMAX"), html.Td(f"{mape_model1:.2f}%")]),
                html.Tr([html.Td("Mean Squared Error (MSE) - XGBoost"), html.Td(f"{xgboost_mse:.2f}")]),
                html.Tr([html.Td("Mean Absolute Error (MAE) - XGBoost"), html.Td(f"{xgboost_mae:.2f}")]),
                html.Tr([html.Td("R^2 Score - XGBoost"), html.Td(f"{xgboost_r2:.2f}")]),
            ])
            
            # MACD plot
            macd_plot_figure = {
                'data': [
                    go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'),
                    go.Scatter(x=df['Date'], y=df['MACD_Signal'], mode='lines', name='MACD Signal', line={'dash': 'dash'})
                ],
                'layout': go.Layout(
                    title='MACD and MACD Signal Line',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Value'}
                )
            }

            # RSI plot
            rsi_plot_figure = {
                'data': [
                    go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='royalblue'))
                ],
                'layout': go.Layout(
                    title='Relative Strength Index (RSI)',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'RSI'},
                    shapes=[
                        dict(
                            type='line',
                            x0=df['Date'].min(),
                            x1=df['Date'].max(),
                            y0=70,
                            y1=70,
                            line=dict(dash='dash', color='red')
                        ),
                        dict(
                            type='line',
                            x0=df['Date'].min(),
                            x1=df['Date'].max(),
                            y0=30,
                            y1=30,
                            line=dict(dash='dash', color='green')
                        )
                    ]
                )
            }
        else:
            price_plot_figure = {}
            xgboost_plot_figure = {}
            metrics_table = ""
            macd_plot_figure = {}
            rsi_plot_figure = {}

        return (price_plot_figure, 
                results_df_sarimax.to_dict('records'),
                [{'name': i, 'id': i} for i in results_df_sarimax.columns],
                xgboost_plot_figure,
                results_df_xgboost.to_dict('records'),
                [{'name': i, 'id': i} for i in results_df_xgboost.columns],
                metrics_table,
                macd_plot_figure,
                rsi_plot_figure)

    return {}, [], [], {}, [], [], "", {}, {}  # Return empty values if no content is provided

# Run the app (it is in local machine)
if __name__ == '__main__':
    app.run_server(mode='external', host="127.0.0.1", port=8056)  # Use a different port number


# In[ ]:




