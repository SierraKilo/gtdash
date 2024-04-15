# imports
import pickle
from dash import Dash, dcc, html, Input, Output, State, dash_table
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
# import os
import base64
# from flask import Flask, request
# from google.colab import drive
# drive.mount('/content/drive')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, assets_folder='assets', assets_url_path='/assets/') #initialising Dash application

# path = '/content/drive/MyDrive/Colab Notebooks/Project/'
path = '/home/shubham/Desktop/Project/Practice/assets/'
# path = './assets/'
model_file = "project_model_rf.pkl"
# UPLOAD_DIRECTORY = './upload_files/'
# if not os.path.exists(UPLOAD_DIRECTORY):
#     os.makedirs(UPLOAD_DIRECTORY)

# load model
loaded_model = pickle.load(open(path+model_file, "rb"))

input_range_low = 20    
input_range_high = 7
temp_range_low = 24
temp_range_high = 35
input_cols = ['FCU_POS', 'INLET_AIR_TEMP']
output_cols = ['S_RPM', 'PT_RPM','OFF_TAKE_TEMP', 'AFTER_HPT_TEMP', 'AIR_PR_AFTER_HPC','FIELD_TEMP_AVG']
input_style={
        'textAlign': 'left',
        'color': '#000000',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '18px',
        'fontWeight': 'bold',
        'fontStyle': 'normal',
        'backgroundColor': '#ffffff',
    }
app.layout = html.Div([
    html.Link(
        rel='stylesheet',
        href='/assets/ess.css'  # Path to your external CSS file
    ),
    html.Div(
        children=[
            html.H1('Gas Turbine Modelling Dashboard', 
                    style={'color': '#ff0000',
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '40px',
                    'fontWeight': 'bold',
                    'fontStyle': 'normal',
                    'textDecoration': 'underline',
                    'textShadow': '2px 2px 2px rgba(0, 0, 0, 0.5)',
                    'padding': '10px', 'margin': '10px'}),
            
            html.H1('Maximizing Gas Turbine Efficiency with Advanced Analytics',
                    style={'color': '#ff0000',
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '28px',
                    'fontWeight': 'bold',
                    'fontStyle': 'normal',
                    'textDecoration': 'underline',
                    'textShadow': '2px 2px 2px rgba(0, 0, 0, 0.5)',
                    'padding': '10px', 'margin': '10px'}),
        ],
        style={
            'width': '100%',
            'height': '100vh',  # vh stands for viewport height
            'background-image': 'url("/assets/iisc_cds.JPEG")',
            'background-size': 'cover',
            'background-repeat': 'no-repeat',
            # 'background-position': 'center center',
            'display': 'flex',
            'flex-direction': 'column',
            'align-items': 'center',
            'justify-content': 'flex-start',
            'text-align': 'center'
        }
    ),
    # html.H1(children = ["Gas Turbine Modelling Dashboard", 
    #                     html.Br(), html.Br(),
    #                     "Maximizing Gas Turbine Efficiency with Advanced Analytics"],
    #         style = {
    #             'textAlign': 'center',
    #             'color': '#ff0000',
    #             'fontFamily': 'Arial, sans-serif',
    #             'fontSize': '36px',
    #             'fontWeight': 'bold',
    #             'fontStyle': 'normal',
    #             'textDecoration': 'underline',
    #             'textShadow': '2px 2px 2px rgba(0, 0, 0, 0.5)',
    #             'backgroundColor': '#f0f0f0',
    #             'lineHeight': '1.2',
    #             # 'background-image': 'url("/assets/iisc_cds.JPEG")',
    #             # 'background-size': 'cover',
    #             # 'background-repeat': 'no-repeat',
    #             # 'background-position': 'center center',
    #             # 'height': '1440px'
    #         }
    #        ),
    html.H3("Gas Turbine Modeling Dashboard leverages cutting-edge machine learning to provide real-time predictions and evaluations of gas turbine performance and health. Our dashboard enables operators and engineers to optimize gas turbine operations, anticipate maintenance needs, and ensure peak efficiency, ultimately saving time and money.",
           style = {
                'textAlign': 'left',
                'color': '#000000',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '12  px',
                # 'fontWeight': 'bold',
                'fontStyle': 'normal',
                # 'textDecoration': 'underline',
                'textShadow': '1px 1px 1px rgba(0, 0, 0, 0.5)',
                'backgroundColor': '#ffffff',
                'lineHeight': '1',
            }
           ),
    html.H2("Select Mode:",
           style = {
                'textAlign': 'left',
                'color': '#000000',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '28px',
                'fontWeight': 'bold',
                'fontStyle': 'normal',
                'textDecoration': 'underline',
                'textShadow': '1px 1px 1px rgba(0, 0, 0, 0.5)',
                'backgroundColor': '#ffffff',
                'lineHeight': '1',
            }
           ),
    dcc.Tabs(id='mode-tabs', value='Individual Cases Mode', children=[
        dcc.Tab(label='Individual Cases Mode', value='Individual Cases Mode', className='tab',
                selected_style={'background-color': '#3498db',
                                'color': 'white',
                                'font-size': '18px',
                                'fontWeight': 'bold',
#                                 'border': '1px solid #3498db',
                                'border-radius': '10px',
                                'cursor': 'pointer',
#                                 'padding': '5px',
#                                 'margin': '5px'
                                'transform': 'scale(1.2)'                                
                               },
                children=[
            html.H3("\nThe model will simulate the gas turbine for increasing fuel flow values at constant ambient conditions.\n",
                    style = {
                        'textAlign': 'left',
                        'color': '#000000',
                        'fontFamily': 'Arial, sans-serif',
                        'fontSize': '18px',
                        'fontStyle': 'normal',
                        'backgroundColor': '#ffffff',
            }),
            html.H3("\nEnter Ambient Temperature (K):",
                    style=input_style
                   ),
            dcc.Slider(
                id='ambient-temperature-input',
                min=temp_range_low,
                max=temp_range_high,
                step=0.5,  # You can adjust the step size as needed
                value=25,  # Initial value
                marks={i: str(i) for i in range(temp_range_low, temp_range_high, 5)},  # You can customize the marks
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            # html.H3("Enter Ambient Pressure (kg/cc):",
            #         style=input_style
            #        ),
            # dcc.Slider(
            #     id='ambient-pressure-input',
            #     min=1.018,
            #     max=1.033,
            #     step=0.001,  # You can adjust the step size as needed
            #     value=1.025,  # Initial value
            #     marks={i / 1000: str(i / 1000) for i in range(1018, 1033, 1)},  # You can customize the marks
            #     tooltip={'placement': 'bottom', 'always_visible': True}
            # ),
            html.Button('Submit', id='submit-button-individual', n_clicks=0, className='button invert'),
            ],
        ),
        dcc.Tab(label='Tent Curve Mode', value='Tent Curve Mode', className='tab',
                selected_style={'background-color': '#3498db',
                                'color': 'white',
                                'font-size': '18px',
                                'fontWeight': 'bold',
#                                 'border': '1px solid #3498db',
                                'border-radius': '10px',
                                'cursor': 'pointer',
#                                 'padding': '5px',
#                                 'margin': '5px',
                                'transform': 'scale(1.2)'                               
                               },
                children=[
            html.H3("\nThe model will simulate the gas turbine for constant inputs at varying ambient temperatures.\n",
                    style = {
                        'textAlign': 'left',
                        'color': '#000000',
                        'fontFamily': 'Arial, sans-serif',
                        'fontSize': '18px',
                        'fontStyle': 'normal',
                        'backgroundColor': '#ffffff',
            }),
            html.H3("\nEnter input FCU value:",
                    style=input_style
                   ),
            dcc.Slider(
                id='fcu-input',
                min=input_range_low,
                max=input_range_high,
                step=1,  # Adjust the step as needed
                value=40,  # Set an initial value
                marks={i / 100: str(i / 100) for i in range(1, 70, 5)},  # Add marks for specific values
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            # html.H2("Enter Ambient Pressure (kg/cc):",
            #         style=input_style
            #        ),
            # dcc.Slider(
            #     id='tent-pressure-input',
            #     min=1.018,
            #     max=1.033,
            #     step=0.001,  # You can adjust the step size as needed
            #     value=1.025,  # Initial value
            #     marks={i / 1000: str(i / 1000) for i in range(1018, 1033, 1)},  # You can customize the marks
            #     tooltip={'placement': 'bottom', 'always_visible': True}
            # ),
            html.H3("Enter value of \u0394t:",
                    style=input_style
                   ),
            dcc.Slider(
                id='delt-input',
                min=0,
                max=16,
                step=1,  # Adjust the step as needed
                value=2,  # Set an initial value
                marks={i: str(i) for i in range(0, 16, 2)},  # Add marks for specific values
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            html.Button('Submit', id='submit-button-tent-curve', n_clicks=0, className='button invert'),
        ],
        ),
        dcc.Tab(label='E-log Mode', value='E-log Mode', className='tab',
                selected_style={'background-color': '#3498db',
                                'color': 'white',
                                'font-size': '18px',
                                'fontWeight': 'bold',
#                                 'border': '1px solid #3498db',
                                'border-radius': '10px',
                                'cursor': 'pointer',
#                                 'padding': '5px',
#                                 'margin': '5px'
                                'transform': 'scale(1.2)'                                
                               },
                children=[
            html.H3("\nThe model will simulate the electronic logs of the gas turbine and check for deviated results.\n",
                    style = {
                        'textAlign': 'left',
                        'color': '#000000',
                        'fontFamily': 'Arial, sans-serif',
                        'fontSize': '18px',
                        'fontStyle': 'normal',
                        'backgroundColor': '#ffffff',
            }),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select e-Log Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='output-data-upload'),
        ],
        ),
    ],
    className='custom-tabs'
    ),
    dcc.Graph(id="graph",
#              figure={
#                  'layout':{
#                      'plot_bgcolor': 'lightgray',  # Background color
#                      'paper_bgcolor': 'white',  # Plot area background color
#                  }
#              }
             ),
    dash_table.DataTable(id='table',
                         style_table={
                             'textAlign': 'left',
                             'font-family': 'Arial, sans-serif',
                             'font-size': '14px',
                             'backgroundColor': 'lightgray',
                             'color': 'black',
                             'border': '1px solid black'},
                         style_cell={'textAlign': 'left'})
],  
#     style={
#         'backgroundColor': '#ffffff',
#         'background-repeat': 'no-repeat',
#         'background-position': 'center',
# }
)


@app.callback(
    Output('graph', 'figure'),
    Output('table', 'data'),
    Input('mode-tabs', 'value'),
    Input('submit-button-individual', 'n_clicks'),
    Input('ambient-temperature-input', 'n_submit'),  # Listen for Enter key press on input fields
    # Input('ambient-pressure-input', 'n_submit'),
    Input('submit-button-tent-curve', 'n_clicks'),
    Input('fcu-input', 'n_submit'),  # Listen for Enter key press on input fields
    # Input('tent-pressure-input', 'n_submit'),
    Input('delt-input', 'n_submit'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('ambient-temperature-input', 'value'),
    # State('ambient-pressure-input', 'value'),
    State('fcu-input', 'value'),
    # State('tent-pressure-input', 'value'),
    State('delt-input', 'value'),   
)
def test_and_display(selected_mode, n_clicks_individual, n_submit_temp, n_clicks_tent_curve, n_submit_fcu, n_submit_dt, contents, filename, amb_temp, fcu_pos, delta_t):
#     if n_clicks > 0:
    test_ex = pd.DataFrame(columns = input_cols)
    final_results = pd.DataFrame(columns = input_cols+output_cols)
    
    if selected_mode == "Individual Cases Mode":
        fcu_pos = np.round(np.linspace(input_range_low, input_range_high, 20), decimals=2)
        test_ex['FCU_POS'] = fcu_pos
        test_ex['INLET_AIR_TEMP'].fillna(amb_temp, inplace=True, limit=20)

        final_results['FCU_POS'] = test_ex['FCU_POS']
        final_results['INLET_AIR_TEMP'] = test_ex['INLET_AIR_TEMP']

        # Scaling the model input data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(test_ex)
        test_norm = scaler.transform(test_ex)
        case1_pred = loaded_model.predict(test_norm)

        final_results['S_RPM'] = case1_pred[:, 0]
        final_results['PT_RPM'] = case1_pred[:, 1]
        final_results['OFF_TAKE_TEMP'] = case1_pred[:, 2]
        final_results['AFTER_HPT_TEMP'] = case1_pred[:, 3]
        final_results['AIR_PR_AFTER_HPC'] = case1_pred[:, 4]
        final_results['FIELD_TEMP_AVG'] = case1_pred[:, 5]

        final_results[output_cols] = final_results[output_cols].apply(pd.to_numeric, errors='ignore')
        
        fig = make_subplots(rows=2, cols=3,
                            shared_xaxes=True,
                            vertical_spacing=0.05)
        
        fig.add_trace(row=1, col=1,
                    trace=go.Scatter(x=final_results['FCU_POS'], y=final_results['S_RPM'],
                              name='S_RPM', mode='lines+markers'))
        fig.add_trace(row=1, col=2,
                    trace=go.Scatter(x=final_results['FCU_POS'], y=final_results['PT_RPM'],
                              name='PT_RPM', mode='lines+markers'))
        fig.add_trace(row=1, col=3,
                    trace=go.Scatter(x=final_results['FCU_POS'], y=final_results['OFF_TAKE_TEMP'],
                              name='OFF_TAKE_TEMP', mode='lines+markers'))
        fig.add_trace(row=2, col=1,
                    trace=go.Scatter(x=final_results['FCU_POS'], y=final_results['AFTER_HPT_TEMP'],
                              name='AFTER_HPT_TEMP', mode='lines+markers'))
        fig.add_trace(row=2, col=2,
                    trace=go.Scatter(x=final_results['FCU_POS'], y=final_results['AIR_PR_AFTER_HPC'],
                              name='AIR_PR_AFTER_HPC', mode='lines+markers'))
        fig.add_trace(row=2, col=3,
                    trace=go.Scatter(x=final_results['FCU_POS'], y=final_results['FIELD_TEMP_AVG'],
                              name='FIELD_TEMP_AVG', mode='lines+markers'))
                
        fig.add_layout_image(
            source="/assets/IISc_Logo-1.jpg",  # Replace with the URL of your watermark image
            x=0,  # Set the x-coordinate for the image (0 is left)
            y=1,  # Set the y-coordinate for the image (0 is bottom)
            xref="paper",  # Use "paper" as the reference for x-coordinate
            yref="paper",  # Use "paper" as the reference for y-coordinate
            sizex=0.1,  # Set the width of the image (1 means 100% of the plot width)
            sizey=0.1,  # Set the height of the image (1 means 100% of the plot height)
            opacity=0.2  # Set the opacity of the watermark (adjust as needed)
        )
        fig.update_layout(
            height=900,
            width=1400,
            title={
                'text': f'<span style="text-decoration: underline;"><b>{selected_mode} - Input Power vs Output Parameters</b>',
                'x': 0.5,  # Adjust the title's horizontal position as needed
                'xanchor': 'center',  # Center the title horizontally
                'font': {
                    'size': 18,  # Adjust the font size as needed
                    'family': 'Arial',  # Specify the font family if needed
                }
            }
        )
        fig.update_yaxes(title_text="S_RPM", row=1, col=1)
        fig.update_yaxes(title_text="PT_RPM", row=1, col=2)
        fig.update_yaxes(title_text="OFF_TAKE_TEMP", row=1, col=3)
        fig.update_yaxes(title_text="AFTER_HPT_TEMP", row=2, col=1)
        fig.update_yaxes(title_text="AIR_PR_AFTER_HPC", row=2, col=2)
        fig.update_yaxes(title_text="FIELD_TEMP_AVG", row=2, col=3)

        fig.add_annotation( text="Input_Power",
                            x=0.5, y=-0.1,
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            font=dict(size=18)
                            )
        # final_results.info()
        final_results = final_results.applymap(lambda x: round(x,3))
        final_results[['S_RPM', 'PT_RPM']]=final_results[['S_RPM', 'PT_RPM']].round()
        # display(final_results)
        table_data = final_results.to_dict('records')
        return fig, table_data

    elif selected_mode == "Tent Curve Mode":
        amb_temp = np.arange(temp_range_low, temp_range_high, delta_t)
        steps = len(amb_temp)

        test_ex['INLET_AIR_TEMP'] = amb_temp
        test_ex['FCU_POS'].fillna(fcu_pos, inplace=True, limit=steps)

        final_results['FCU_POS'] = test_ex['FCU_POS']
        final_results['INLET_AIR_TEMP'] = test_ex['INLET_AIR_TEMP']

        # Scaling the model input data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(test_ex)
        test_norm = scaler.transform(test_ex)
        case2_pred = loaded_model.predict(test_norm)

        final_results['S_RPM'] = case2_pred[:, 0]
        final_results['PT_RPM'] = case2_pred[:, 1]
        final_results['OFF_TAKE_TEMP'] = case2_pred[:, 2]
        final_results['AFTER_HPT_TEMP'] = case2_pred[:, 3]
        final_results['AIR_PR_AFTER_HPC'] = case2_pred[:, 4]
        final_results['FIELD_TEMP_AVG'] = case2_pred[:, 5]

        fig = make_subplots(rows=2, cols=3,
                            shared_xaxes=True,
                            vertical_spacing=0.05)
        
        fig.add_trace(row=1, col=1,
                    trace=go.Scatter(x=final_results['INLET_AIR_TEMP'], y=final_results['S_RPM'],
                              name='S_RPM', mode='lines+markers'))
        fig.add_trace(row=1, col=2,
                    trace=go.Scatter(x=final_results['INLET_AIR_TEMP'], y=final_results['PT_RPM'],
                              name='PT_RPM', mode='lines+markers'))
        fig.add_trace(row=1, col=3,
                    trace=go.Scatter(x=final_results['INLET_AIR_TEMP'], y=final_results['OFF_TAKE_TEMP'],
                              name='OFF_TAKE_TEMP', mode='lines+markers'))
        fig.add_trace(row=2, col=1,
                    trace=go.Scatter(x=final_results['INLET_AIR_TEMP'], y=final_results['AFTER_HPT_TEMP'],
                              name='AFTER_HPT_TEMP', mode='lines+markers'))
        fig.add_trace(row=2, col=2,
                    trace=go.Scatter(x=final_results['INLET_AIR_TEMP'], y=final_results['AIR_PR_AFTER_HPC'],
                              name='AIR_PR_AFTER_HPC', mode='lines+markers'))
        fig.add_trace(row=2, col=3,
                    trace=go.Scatter(x=final_results['INLET_AIR_TEMP'], y=final_results['FIELD_TEMP_AVG'],
                              name='FIELD_TEMP_AVG', mode='lines+markers'))
        
        fig.add_layout_image(
            source="/assets/IISc_Logo-1.jpg",  # Replace with the URL of your watermark image
            x=0,  # Set the x-coordinate for the image (0 is left)
            y=1,  # Set the y-coordinate for the image (0 is bottom)
            xref="paper",  # Use "paper" as the reference for x-coordinate
            yref="paper",  # Use "paper" as the reference for y-coordinate
            sizex=0.1,  # Set the width of the image (1 means 100% of the plot width)
            sizey=0.1,  # Set the height of the image (1 means 100% of the plot height)
            opacity=0.2  # Set the opacity of the watermark (adjust as needed)
        )
        fig.update_layout(
            height=900,
            width=1400,
            title={
                'text': f'<span style="text-decoration: underline;"><b>{selected_mode} - Ambient Temperature vs Output Parameters</b>',
                'x': 0.5,  # Adjust the title's horizontal position as needed
                'xanchor': 'center',  # Center the title horizontally
                'font': {
                    'size': 18,  # Adjust the font size as needed
                    'family': 'Arial',  # Specify the font family if needed
                }
            }
        )

        fig.update_yaxes(title_text="S_RPM", row=1, col=1)
        fig.update_yaxes(title_text="PT_RPM", row=1, col=2)
        fig.update_yaxes(title_text="OFF_TAKE_TEMP", row=1, col=3)
        fig.update_yaxes(title_text="AFTER_HPT_TEMP", row=2, col=1)
        fig.update_yaxes(title_text="AIR_PR_AFTER_HPC", row=2, col=2)
        fig.update_yaxes(title_text="FIELD_TEMP_AVG", row=2, col=3)

        fig.add_annotation(
                            text="Ambient Temperature",
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=-0.1,
                            showarrow=False,
                            font=dict(size=18)
                            )
        # final_results.info()
        final_results = final_results.applymap(lambda x: round(x,3))
        final_results[['S_RPM', 'PT_RPM']]=final_results[['S_RPM', 'PT_RPM']].round()

        
        # display(final_results)
        table_data = final_results.to_dict('records')
        return fig, table_data

    elif selected_mode == "E-log Mode":

        # full_path = os.path.join(UPLOAD_DIRECTORY, filename)
        # with open(full_path, 'wb') as f:
        #     f.write(contents.encode("utf8"))

        # Check if files were uploaded
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(filename, sep = ',', engine = 'python')
            elif 'xls' in filename or 'xlsx' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(filename)
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        
        df.dropna(axis=1,how='all',inplace=True)
        columns = ['DTG', 'ENGINE_NO', 'REAL_POWER', 'FCU_POS', 'S_RPM', 'LPC_RPM',
       'HPC_RPM', 'PT_RPM', 'INLET_AIR_TEMP', 'OFF_TAKE_TEMP',
       'AFTER_HPT_TEMP', 'AIR_PR_AFTER_HPC', 'FUEL_PRESS_CHANNEL1',
       'FUEL_PRESS_CHANNEL2', 'FIELD_TEMP1', 'FIELD_TEMP2', 'FIELD_TEMP3',
       'FIELD_TEMP4', 'FIELD_TEMP5', 'FIELD_TEMP6', 'FIELD_TEMP7',
       'FIELD_TEMP8', 'FIELD_TEMP9', 'FIELD_TEMP10', 'FIELD_TEMP_AVG',
       'FIELD_TEMP_HIGH', 'FIELD_TEMP_LOW', 'VIB_LPC', 'VIB_PT', 'TOTAL_RH']
        if None in df.columns:
            df.columns = columns
        required_cols = input_cols + output_cols
        
        X = df[input_cols]
        y = df[output_cols]

        # Scaling the model input data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X)

        case3_pred = loaded_model.predict(X_norm)
        fig = make_subplots(rows=2, cols=3, shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(row=1, col=1, trace=go.Scatter(x=X['FCU_POS'], y=y['S_RPM'],
                            name='S_RPM', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=1, col=2, trace=go.Scatter(x=X['FCU_POS'], y=y['PT_RPM'],
                            name='PT_RPM', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=1, col=3, trace=go.Scatter(x=X['FCU_POS'], y=y['OFF_TAKE_TEMP'],
                            name='OFF_TAKE_TEMP', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=2, col=1, trace=go.Scatter(x=X['FCU_POS'], y=y['AFTER_HPT_TEMP'],
                            name='AFTER_HPT_TEMP', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=2, col=2, trace=go.Scatter(x=X['FCU_POS'], y=y['AIR_PR_AFTER_HPC'],
                            name='AIR_PR_AFTER_HPC', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=2, col=3, trace=go.Scatter(x=X['FCU_POS'], y=y['FIELD_TEMP_AVG'],
                            name='FIELD_TEMP_AVG', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=1, col=1, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,0],
                                name='Predicited S_RPM', mode='markers', line=dict(color='red')))
        fig.add_trace(row=1, col=2, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,1],
                                name='Predicted PT_RPM', mode='markers', line=dict(color='red')))
        fig.add_trace(row=1, col=3, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,2],
                                name='Predicted OFF_TAKE_TEMP', mode='markers', line=dict(color='red')))
        fig.add_trace(row=2, col=1, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,3],
                                name='Predicted AFTER_HPT_TEMP', mode='markers', line=dict(color='red')))
        fig.add_trace(row=2, col=2, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,4],
                                name='Predicted AIR_PR_AFTER_HPC', mode='markers', line=dict(color='red')))
        fig.add_trace(row=2, col=3, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,5],
                                name='Predicted FIELD_TEMP_AVG', mode='markers', line=dict(color='red')))
        
        fig.add_layout_image(
            source="/assets/IISc_Logo-1.jpg",  # Replace with the URL of your watermark image
            x=0,  # Set the x-coordinate for the image (0 is left)
            y=1,  # Set the y-coordinate for the image (0 is bottom)
            xref="paper",  # Use "paper" as the reference for x-coordinate
            yref="paper",  # Use "paper" as the reference for y-coordinate
            sizex=0.1,  # Set the width of the image (1 means 100% of the plot width)
            sizey=0.1,  # Set the height of the image (1 means 100% of the plot height)
            opacity=0.2  # Set the opacity of the watermark (adjust as needed)
        )
        
        fig.update_layout(
            height=900,
            width=1400,
            title={
                'text': f'<span style="text-decoration: underline;"><b>{selected_mode} - FCU_POS vs Output Parameters</b>',
                'x': 0.5,  # Adjust the title's horizontal position as needed
                'xanchor': 'center',  # Center the title horizontally
                'font': {
                    'size': 18,  # Adjust the font size as needed
                    'family': 'Arial',  # Specify the font family if needed
                }
            }
        )
        fig.update_yaxes(title_text="S_RPM", row=1, col=1)
        fig.update_yaxes(title_text="PT_RPM", row=1, col=2)
        fig.update_yaxes(title_text="OFF_TAKE_TEMP", row=1, col=3)
        fig.update_yaxes(title_text="AFTER_HPT_TEMP", row=2, col=1)
        fig.update_yaxes(title_text="AIR_PR_AFTER_HPC", row=2, col=2)
        fig.update_yaxes(title_text="FIELD_TEMP_AVG", row=2, col=3)

        fig.add_annotation( text="FCU_POS",
                            x=0.5, y=-0.1,
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            font=dict(size=18)
                            )

        # Checking for out of limit cases
        j,count = 0,0
        alerts = pd.DataFrame(columns=['SER','ALERTS'])
        for i, row in y.iterrows():
            if y.iloc[j]['AFTER_HPT_TEMP'] > 1.05 * case3_pred[j][3]:
                message = f'Parameter out of predicted limit at index: {i}'
                count+=1
                new_alert = pd.DataFrame({'SER': count, 'ALERTS': [message]})
                alerts = pd.concat([alerts, new_alert], ignore_index=True)                
            j+=1
        message = f'Total out of limit instances: {count}'
        alerts = pd.concat([alerts, pd.DataFrame({'ALERTS': [message]})], ignore_index=True)
        table_data = alerts.to_dict('records')

        # os.remove(full_path)
        return fig, table_data
    

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)