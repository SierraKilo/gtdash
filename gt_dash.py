# imports
import pickle
import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap 
import seaborn as sns
from PIL import Image
import io
# import os
import base64
import csv
# from flask import Flask, request
# from google.colab import drive
# drive.mount('/content/drive')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, assets_folder='assets', assets_url_path='/assets/', external_stylesheets=external_stylesheets) #initialising Dash application

path = '/Users/shubham/Desktop/IISc/MTech_Project/Practice/assets/'
# path = '/home/shubham/Desktop/Project/Practice/assets/'
# path = './assets/'
model_file = "rf_model1.pkl"
# UPLOAD_DIRECTORY = './upload_files/'
# if not os.path.exists(UPLOAD_DIRECTORY):
#     os.makedirs(UPLOAD_DIRECTORY)

# load model
loaded_model = pickle.load(open(path+model_file, "rb"))

input_range_low = 20    
input_range_high = 100
temp_range_low = 24
temp_range_high = 35
input_cols = ['FCU_POS', 'INLET_AIR_TEMP']
# output_cols = ['S_RPM', 'PT_RPM','OFF_TAKE_TEMP', 'AFTER_HPT_TEMP', 'AIR_PR_AFTER_HPC','FIELD_TEMP_AVG']
output_cols = ['LPC_RPM', 'HPC_RPM', 'AIR_PR_AFTER_HPC', 'FIELD_TEMP_AVG', 'AFTER_HPT_TEMP', 'OFF_TAKE_TEMP', 'PT_RPM', 'S_RPM']

input_style={
        'textAlign': 'left',
        'color': '#000000',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '18px',
        'fontWeight': 'bold',
        'fontStyle': 'normal',
        'backgroundColor': '#ffffff',
        'marginBottom': 10
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
            
            # html.Button('Submit', id='submit-button-individual', n_clicks=0, className='button invert'),

            dcc.Graph(id="graph-icm",
            ),
            html.Button("Download as CSV", id="btn_csv_tab1", n_clicks=0, className='button invert'),
            dcc.Download(id="download-df-csv-icm"),
            dash_table.DataTable(id='table-icm',
                         style_table={
                             'textAlign': 'left',
                             'font-family': 'Arial, sans-serif',
                             'font-size': '14px',
                             'backgroundColor': 'lightgray',
                             'color': 'black',
                             'border': '1px solid black'},
                         style_cell={'textAlign': 'left'}),
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
                marks={i: str(i) for i in range(20, 100, 5)},  # Add marks for specific values
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            
            html.H3("Enter value of \u0394t:",
                    style=input_style
                   ),
            dcc.Slider(
                id='delt-input',
                min=1,
                max=16,
                step=1,  # Adjust the step as needed
                value=2,  # Set an initial value
                marks={i: str(i) for i in range(0, 16, 2)},  # Add marks for specific values
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            # html.Button('Submit', id='submit-button-tent-curve', n_clicks=0, className='button invert'),

            dcc.Graph(id="graph-tcm",
            ),
            html.Button("Download as CSV", id="btn_csv_tab2", n_clicks=0, className='button invert'),
            dcc.Download(id="download-df-csv-tcm"),
            dash_table.DataTable(id='table-tcm',
                         style_table={
                             'textAlign': 'left',
                             'font-family': 'Arial, sans-serif',
                             'font-size': '14px',
                             'backgroundColor': 'lightgray',
                             'color': 'black',
                             'border': '1px solid black'},
                         style_cell={'textAlign': 'left'}),
            
        ],
    ),
        dcc.Tab(label='Digital Twin', value='Digital Twin', className='tab',
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
            html.H3("\nThe model will simulate the gas turbine for the specified input power and ambient temperature.\n",
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
                id='fcu-dt-input',
                min=input_range_low,
                max=input_range_high,
                step=1,  # Adjust the step as needed
                value=40,  # Set an initial value
                marks={i: str(i) for i in range(20, 100, 5)},  # Add marks for specific values
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            html.H3("\nEnter Ambient Temperature (K):",
                    style=input_style
                   ),
            dcc.Slider(
                id='amb-temp-input',
                min=temp_range_low,
                max=temp_range_high,
                step=0.5,  # You can adjust the step size as needed
                value=25,  # Initial value
                marks={i: str(i) for i in range(temp_range_low, temp_range_high, 5)},  # You can customize the marks
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            # html.Button('Submit', id='submit-button-digital-twin', n_clicks=0, className='button invert'),

            # html.Img(id="image-dt", src="path_to_your_image"),  # replace 'path_to_your_image' with the actual path to your image
            html.Div(id='image-dt', style={'display': 'flex', 'justify-content': 'center'}),  # This is where the image will be displayed
            html.Button("Download as CSV", id="btn_csv_tab3", n_clicks=0, className='button invert'),
            dcc.Download(id="download-df-csv-dt"),
            dash_table.DataTable(id='table-dt',
                     style_table={
                         'textAlign': 'left',
                         'font-family': 'Arial, sans-serif',
                         'font-size': '14px',
                         'backgroundColor': 'lightgray',
                         'color': 'black',
                         'border': '1px solid black'},
                     style_cell={'textAlign': 'left'}),
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

            dcc.Graph(id="graph-elm",
            ),
            html.Button("Download as CSV", id="btn_csv_tab4", n_clicks=0, className='button invert'),
            dcc.Download(id="download-df-csv-elm"),
            dash_table.DataTable(id='table-elm',
                         style_table={
                             'textAlign': 'left',
                             'font-family': 'Arial, sans-serif',
                             'font-size': '14px',
                             'backgroundColor': 'lightgray',
                             'color': 'black',
                             'border': '1px solid black'},
                         style_cell={'textAlign': 'left'}),

        ],
    ),
    dcc.Tab(label='SLIP Calculator', value='SLIP Calculator', className='tab',
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
            html.H3("\nThe model will calculate slip and give alerts for undertaking steam or chemical washing.\n",
                    style = {
                        'textAlign': 'left',
                        'color': '#000000',
                        'fontFamily': 'Arial, sans-serif',
                        'fontSize': '18px',
                        'fontStyle': 'normal',
                        'backgroundColor': '#ffffff',
            }),
            dbc.Row([
                        dbc.Col([
                            html.H3("\nEnter observed LPC RPM (at 0.66N):", style = input_style),
                            dcc.Input(
                                id='lpc-rpm-input',
                                type='number',
                                value=5500,  # Set an initial value
                                min=1750,
                                max=5750,
                            ),
                        ], xs=6),
                        dbc.Col([
                            html.H3("\nEnter observed HPC RPM (at 0.66N):", style = input_style),
                            dcc.Input(
                                id='hpc-rpm-input',
                                type='number',
                                value=7430,  # Initial value
                                min=3750,
                                max=8000,
                            ),
                        ], xs=6),
                    ], style={'marginBottom': 20}),
            dbc.Row([
                        dbc.Col([
                            html.H3("\nEnter observed Air Pressure after HPC (at 0.66N):", style = input_style),
                            dcc.Input(
                                id='airpr-hpc-input',
                                type='number',
                                value=7.5,  # Set an initial value
                                min=0.4,
                                max=8,
                            ),
                        ], xs=6),
                        dbc.Col([
                            html.H3("\nEnter observed Exhaust Temperature (at 0.66N):", style = input_style),
                            dcc.Input(
                                id='ex-temp-input',
                                type='number',
                                value=500,  # Initial value
                                min=250,
                                max=650,
                            ),
                        ], xs=6),
                    ], style={'marginBottom': 20}),
                    html.Button('Submit', id='submit-button-slip', n_clicks=0, className='button invert'),

            dcc.Graph(id="graph-slip",
            ),
            dash_table.DataTable(id='table-slip',
                         style_table={
                             'textAlign': 'left',
                             'font-family': 'Arial, sans-serif',
                             'font-size': '14px',
                             'backgroundColor': 'lightgray',
                             'color': 'black',
                             'border': '1px solid black'},
                         style_cell={'textAlign': 'left'}),
        ],
    ),
        
    ],
    className='custom-tabs'
    ),

])

@app.callback(
    [Output('graph-icm', 'figure'),
     Output('table-icm', 'data'),
     Output("download-df-csv-icm", "data"),
     Output('graph-tcm', 'figure'),
     Output('table-tcm', 'data'),
     Output("download-df-csv-tcm", "data"),
     Output('image-dt', 'children'),
     Output('table-dt', 'data'),
     Output("download-df-csv-dt", "data"),
     Output('graph-elm', 'figure'),
     Output('table-elm', 'data'),
     Output("download-df-csv-elm", "data"),
     Output('graph-slip', 'figure'),
     Output('table-slip', 'data')],
    [Input('mode-tabs', 'value'),
     # Input('submit-button-individual', 'n_clicks'),
     Input('ambient-temperature-input', 'value'),  
     # Input('submit-button-tent-curve', 'n_clicks'),
     Input('fcu-input', 'value'), 
     Input('delt-input', 'value'),
     # Input('submit-button-digital-twin', 'n_clicks'),
     Input('fcu-dt-input', 'value'), 
     Input('amb-temp-input', 'value'),
     Input('upload-data', 'contents'),
    #  Input('lpc-rpm-input', 'value'), 
    #  Input('hpc-rpm-input', 'value'),
    #  Input('airpr-hpc-input', 'value'), 
    #  Input('ex-temp-input', 'value'),
     Input('submit-button-slip', 'n_clicks'),
     Input("btn_csv_tab1", "n_clicks"),
     Input("btn_csv_tab2", "n_clicks"),
     Input("btn_csv_tab3", "n_clicks"),
     Input("btn_csv_tab4", "n_clicks")],
    [State('upload-data', 'filename'),
     State('lpc-rpm-input', 'value'),
     State('hpc-rpm-input', 'value'),
     State('airpr-hpc-input', 'value'),
     State('ex-temp-input', 'value')], 
    # State('amb-temp-input', 'value'),
    # prevent_initial_call=True,     
)
def test_and_display(selected_mode, amb_temp, fcu_pos, delta_t, fcu_dt, amb_temp_dt, contents, n_clicks_submit, n_clicks_tab1, n_clicks_tab2, n_clicks_tab3, n_clicks_tab4, filename, lpc_rpm, hpc_rpm, airpr_hpc, ex_temp):
# def test_and_display(selected_mode, contents, n_clicks_table, filename, amb_temp, fcu_pos, delta_t, fcu_dt, amb_temp_dt):
    
    # test_ex = pd.DataFrame(columns = input_cols)
    # final_results = pd.DataFrame(columns = input_cols+output_cols)
    
    if selected_mode == "Individual Cases Mode":
        return ICMode(selected_mode, amb_temp, n_clicks_tab1)
    elif selected_mode == "Tent Curve Mode":
        return TCMode(selected_mode, fcu_pos, delta_t, n_clicks_tab2)
    elif selected_mode == "Digital Twin":
        return DTMode(selected_mode, fcu_dt, amb_temp_dt, n_clicks_tab3)     
    elif selected_mode == "E-log Mode":
        return ELMode(selected_mode, contents, n_clicks_tab4, filename)
    elif selected_mode == "SLIP Calculator":
        return Slip(selected_mode, lpc_rpm, hpc_rpm, airpr_hpc, ex_temp, n_clicks_submit)


def ICMode(selected_mode, amb_temp, n_clicks_tab1):

    test_ex = pd.DataFrame(columns = input_cols)
    final_results = pd.DataFrame(columns = input_cols+output_cols)
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
    case1_pred = loaded_model.predict(test_ex)

    final_results['LPC_RPM'] = case1_pred[:, 0]
    final_results['HPC_RPM'] = case1_pred[:, 1]
    final_results['AIR_PR_AFTER_HPC'] = case1_pred[:, 2]
    final_results['FIELD_TEMP_AVG'] = case1_pred[:, 3]
    final_results['AFTER_HPT_TEMP'] = case1_pred[:, 4]
    final_results['OFF_TAKE_TEMP'] = case1_pred[:, 5]
    final_results['PT_RPM'] = case1_pred[:, 6]
    final_results['S_RPM'] = case1_pred[:, 7]

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
    final_results = final_results.applymap(lambda x: round(x,3))
    final_results[['LPC_RPM', 'HPC_RPM', 'S_RPM', 'PT_RPM']]=final_results[['LPC_RPM', 'HPC_RPM', 'S_RPM', 'PT_RPM']].round()
    table_data = final_results.to_dict('records')
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    # print(changed_id)
    if 'btn_csv_tab1.n_clicks' in changed_id:
        # n_clicks_tab1 = None
        return fig, table_data, dcc.send_data_frame(final_results.to_csv, "individual_cases_mode_data.csv"), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    else:
        return fig, table_data, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


def TCMode(selected_mode, fcu_pos, delta_t, n_clicks_tab2):
    test_ex = pd.DataFrame(columns = input_cols)
    final_results = pd.DataFrame(columns = input_cols+output_cols)
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
    case2_pred = loaded_model.predict(test_ex)

    final_results['LPC_RPM'] = case2_pred[:, 0]
    final_results['HPC_RPM'] = case2_pred[:, 1]
    final_results['AIR_PR_AFTER_HPC'] = case2_pred[:, 2]
    final_results['FIELD_TEMP_AVG'] = case2_pred[:, 3]
    final_results['AFTER_HPT_TEMP'] = case2_pred[:, 4]
    final_results['OFF_TAKE_TEMP'] = case2_pred[:, 5]
    final_results['PT_RPM'] = case2_pred[:, 6]
    final_results['S_RPM'] = case2_pred[:, 7]

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
    final_results = final_results.applymap(lambda x: round(x,3))
    final_results[['LPC_RPM', 'HPC_RPM', 'S_RPM', 'PT_RPM']]=final_results[['LPC_RPM', 'HPC_RPM', 'S_RPM', 'PT_RPM']].round()

    
    table_data = final_results.to_dict('records')
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn_csv_tab2.n_clicks' in changed_id:
        return dash.no_update, dash.no_update, dash.no_update, fig, table_data, dcc.send_data_frame(final_results.to_csv, "tent_curve_mode_data.csv"), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    else:
        return dash.no_update, dash.no_update, dash.no_update, fig, table_data, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


def DTMode(selected_mode, fcu_dt, amb_temp_dt, n_clicks_tab3):
    test_ex = pd.DataFrame(columns = input_cols)
    final_results = pd.DataFrame(columns = input_cols+output_cols)
    fcu_pos = np.round(np.linspace(fcu_dt, fcu_dt, 1), decimals=2)
    amb_temp = np.round(np.linspace(amb_temp_dt, amb_temp_dt, 1), decimals=2)

    test_ex['FCU_POS'] = fcu_pos
    test_ex['INLET_AIR_TEMP'] = amb_temp
    # print(fcu_dt, amb_temp_dt)

    final_results['FCU_POS'] = test_ex['FCU_POS']
    final_results['INLET_AIR_TEMP'] = test_ex['INLET_AIR_TEMP']
    # print(test_ex)
    # Scaling the model input data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(test_ex)
    test_norm = scaler.transform(test_ex)
    case4_pred = loaded_model.predict(test_ex)
    
    final_results['LPC_RPM'] = case4_pred[:, 0]
    final_results['HPC_RPM'] = case4_pred[:, 1]
    final_results['AIR_PR_AFTER_HPC'] = case4_pred[:, 2]
    final_results['FIELD_TEMP_AVG'] = case4_pred[:, 3]
    final_results['AFTER_HPT_TEMP'] = case4_pred[:, 4]
    final_results['OFF_TAKE_TEMP'] = case4_pred[:, 5]
    final_results['PT_RPM'] = case4_pred[:, 6]
    final_results['S_RPM'] = case4_pred[:, 7]

    final_results = final_results.applymap(lambda x: round(x,3))
    final_results[['LPC_RPM', 'HPC_RPM', 'S_RPM', 'PT_RPM']]=final_results[['LPC_RPM', 'HPC_RPM', 'S_RPM', 'PT_RPM']].round()

    # Create custom colormaps for red, blue, and black gradients
    red_colors = [(1, 0, 0), (0.5, 0, 0)]  # Start with red, end with dark red
    blue_colors = [(0, 0, 1), (0, 0, 0.5)]  # Start with blue, end with dark blue
    black_colors = [(0, 0, 0), (0.3, 0.3, 0.3)]  # Start with black, end with dark gray

    red_cmap = LinearSegmentedColormap.from_list("red_cmap", red_colors)
    blue_cmap = LinearSegmentedColormap.from_list("blue_cmap", blue_colors)
    black_cmap = LinearSegmentedColormap.from_list("black_cmap", black_colors)

    fig, ax = plt.subplots()

    lpc = patches.Polygon([[1, 1], [1, 3], [2, 2.5], [2, 1.5]], closed=True, fill=True, color=blue_cmap(0.5), edgecolor='black')
    hpc = patches.Polygon([[3, 1.25], [3, 2.75], [3.8, 2.25], [3.8, 1.75]], closed=True, fill=True, color=blue_cmap(0.5), edgecolor='black')
    cc = patches.Polygon([[5, 1.5], [5, 2.5], [5.5, 2.5], [5.5, 1.5]], closed=True, fill=True, color=red_cmap(0.5), edgecolor='r')
    hpt = patches.Polygon([[6.7, 1.75], [6.7, 2.25], [7.5, 2.75], [7.5, 1.25]], closed=True, fill=True, color=red_cmap(0.5), edgecolor='black')
    lpt = patches.Polygon([[8.5, 1.5], [8.5, 2.5], [9.5, 3], [9.5, 1]], closed=True, fill=True, color=red_cmap(0.5), edgecolor='black')
    pt = patches.Polygon([[11, 1.25], [11, 2.75], [12, 3.25], [12, 0.75]], closed=True, fill=True, color=red_cmap(0.5), edgecolor='black')
    sh1 = patches.Polygon([[2, 1.9], [2, 2.1], [8.5, 2.1], [8.5, 1.9]], closed=True, fill=None, edgecolor='black')
    sh2 = patches.Polygon([[12, 1.9], [12, 2.1], [17, 2.1], [17, 1.9]], closed=True, fill=None, edgecolor='black')
    rg = patches.Polygon([[14, 1.5], [14, 2.5], [15, 2.5], [15, 1.5]], closed=True, fill=True, color=black_cmap(0.5), edgecolor='black')
    pp = patches.Polygon([[17, 2], [16.5, 2.25], [17.5, 2.25], [16.5, 1.75], [17.5, 1.75]], closed=True, fill=True, color=black_cmap(0.5), edgecolor='black')

    # Add the trapezoid to the plot
    ax.add_patch(sh1)
    ax.add_patch(lpc)
    ax.add_patch(hpc)
    ax.add_patch(cc)
    ax.add_patch(hpt)
    ax.add_patch(lpt)
    ax.add_patch(sh2)
    ax.add_patch(pt)
    ax.add_patch(rg)
    ax.add_patch(pp)

    # Add labels inside the polygons
    ax.annotate("LPC", (1.5, 2), color='white', weight='bold', fontsize=11, ha='center', va='center', rotation=90)
    ax.annotate("HPC", (3.4, 2), color='white', weight='bold', fontsize=11, ha='center', va='center', rotation=90)
    ax.annotate("CC", (5.3, 2), color='white', weight='bold', fontsize=11, ha='center', va='center', rotation=90)
    ax.annotate("HPT", (7.15, 2), color='white', weight='bold', fontsize=11, ha='center', va='center', rotation=90)
    ax.annotate("LPT", (9, 2), color='white', weight='bold', fontsize=11, ha='center', va='center', rotation=90)
    ax.annotate("PT", (11.5, 2), color='white', weight='bold', fontsize=11, ha='center', va='center', rotation=90)
    ax.annotate("RG", (14.5, 2), color='white', weight='bold', fontsize=11, ha='center', va='center', rotation=90)
    ax.annotate("PROP", (18, 2), color=black_cmap(0.5), weight='bold', fontsize=11, ha='center', va='center', rotation=90)

    LPC_RPM = final_results['LPC_RPM'][0]
    HPC_RPM = final_results['HPC_RPM'][0]
    AIR_PR_AFTER_HPC = final_results['AIR_PR_AFTER_HPC'][0]
    FIELD_TEMP_AVG = final_results['FIELD_TEMP_AVG'][0]
    AFTER_HPT_TEMP = final_results['AFTER_HPT_TEMP'][0]
    OFF_TAKE_TEMP = final_results['OFF_TAKE_TEMP'][0]
    PT_RPM = final_results['PT_RPM'][0]
    S_RPM = final_results['S_RPM'][0]

    # Add text below the polygons with variable values
    ax.annotate("LPC_RPM = ", (1.5, 0), color='black', fontsize=8, ha='center', va='top', rotation=90)
    ax.annotate("{:.2f}".format(LPC_RPM), (1.5, 0), color='black', fontsize=8, weight='bold', ha='center', va='bottom', rotation=90)

    ax.annotate("AIR_PR_AFTER_HPC = ", (3.4, 0), color='black', fontsize=8, ha='center', va='top', rotation=90)
    ax.annotate("{:.2f}".format(AIR_PR_AFTER_HPC), (3.4, 0), color='black', fontsize=8, weight='bold', ha='center', va='bottom', rotation=90)

    ax.annotate("FIELD_TEMP_AVG = ", (5.3, 0), color='black', fontsize=8, ha='center', va='top', rotation=90)
    ax.annotate("{:.2f}".format(FIELD_TEMP_AVG), (5.3, 0), color='black', fontsize=8, weight='bold', ha='center', va='bottom', rotation=90)

    ax.annotate("AFTER_HPT_TEMP = ", (7.15, 0), color='black', fontsize=8, ha='center', va='top', rotation=90)
    ax.annotate("{:.2f}".format(AFTER_HPT_TEMP), (7.15, 0), color='black', fontsize=8, weight='bold', ha='center', va='bottom', rotation=90)

    ax.annotate("OFF_TAKE_TEMP = ", (9, 0), color='black', fontsize=8, ha='center', va='top', rotation=90)
    ax.annotate("{:.2f}".format(OFF_TAKE_TEMP), (9, 0), color='black', fontsize=8, weight='bold', ha='center', va='bottom', rotation=90)

    ax.annotate("PT_RPM = ", (11.5, 0), color='black', fontsize=8, ha='center', va='top', rotation=90)
    ax.annotate("{:.2f}".format(PT_RPM), (11.5, 0), color='black', fontsize=8, weight='bold', ha='center', va='bottom', rotation=90)

    ax.annotate("S_RPM = ", (14.5, 0), color='black', fontsize=8, ha='center', va='top', rotation=90)
    ax.annotate("{:.2f}".format(S_RPM), (14.5, 0), color='black', fontsize=8, weight='bold', ha='center', va='bottom', rotation=90)

    # Set the limits of the plot
    ax.set_xlim([0, 20])
    ax.set_ylim([-3, 5])

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the matplotlib plot as an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Use an HTML img tag to display the image
    image_html = html.Img(src=f'data:image/png;base64,{image_base64}')

    # Add the image to your Plotly figure
    fig = go.Figure()
    fig.add_layout_image(
        source='data:image/png;base64,{}'.format(image_base64),
        x=0,
        y=1,
        xref="paper",
        yref="paper",
        sizex=1,
        sizey=1.1,
        opacity=1,
        layer="below"
    )
            
    table_data = final_results.to_dict('records')
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn_csv_tab3.n_clicks' in changed_id:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, image_html, table_data, dcc.send_data_frame(final_results.to_csv, "digital_twin_data.csv"), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, image_html, table_data, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


def ELMode(selected_mode, contents, n_clicks_tab4, filename):
    test_ex = pd.DataFrame(columns = input_cols)
    final_results = pd.DataFrame(columns = input_cols+output_cols)
    # Check if files were uploaded
    if contents is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename or 'xlsx' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
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
        # X_norm = scaler.fit_transform(X)

        case3_pred = loaded_model.predict(X)
        fig = make_subplots(rows=2, cols=4, shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(row=1, col=1, trace=go.Scatter(x=X['FCU_POS'], y=y['LPC_RPM'],
                            name='LPC_RPM', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=1, col=2, trace=go.Scatter(x=X['FCU_POS'], y=y['HPC_RPM'],
                            name='HPC_RPM', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=1, col=3, trace=go.Scatter(x=X['FCU_POS'], y=y['AIR_PR_AFTER_HPC'],
                            name='AIR_PR_AFTER_HPC', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=1, col=4, trace=go.Scatter(x=X['FCU_POS'], y=y['FIELD_TEMP_AVG'],
                            name='FIELD_TEMP_AVG', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=2, col=1, trace=go.Scatter(x=X['FCU_POS'], y=y['AFTER_HPT_TEMP'],
                            name='AFTER_HPT_TEMP', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=2, col=2, trace=go.Scatter(x=X['FCU_POS'], y=y['OFF_TAKE_TEMP'],
                            name='OFF_TAKE_TEMP', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=2, col=3, trace=go.Scatter(x=X['FCU_POS'], y=y['PT_RPM'],
                            name='PT_RPM', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=2, col=4, trace=go.Scatter(x=X['FCU_POS'], y=y['S_RPM'],
                            name='S_RPM', mode='markers', line=dict(color='blue')))
        fig.add_trace(row=1, col=1, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,0],
                                name='Predicited LPC_RPM', mode='markers', line=dict(color='red')))
        fig.add_trace(row=1, col=2, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,1],
                                name='Predicted HPC_RPM', mode='markers', line=dict(color='red')))
        fig.add_trace(row=1, col=3, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,2],
                                name='Predicted AIR_PR_AFTER_HPC', mode='markers', line=dict(color='red')))
        fig.add_trace(row=1, col=4, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,3],
                                name='Predicted FIELD_TEMP_AVG', mode='markers', line=dict(color='red')))
        fig.add_trace(row=2, col=1, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,4],
                                name='Predicted AFTER_HPT_TEMP', mode='markers', line=dict(color='red')))
        fig.add_trace(row=2, col=2, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,5],
                                name='Predicted OFF_TAKE_TEMP', mode='markers', line=dict(color='red')))
        fig.add_trace(row=2, col=3, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,6],
                                name='Predicted PT_RPM', mode='markers', line=dict(color='red')))
        fig.add_trace(row=2, col=4, trace=go.Scatter(x=X['FCU_POS'], y=case3_pred[:,7],
                                name='Predicted S_RPM', mode='markers', line=dict(color='red')))
        
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
            width=1500,
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
            if y.iloc[j]['AFTER_HPT_TEMP'] > 1.05 * case3_pred[j][4]:
                message = f'Parameter out of predicted limit at index: {i}'
                count+=1
                new_alert = pd.DataFrame({'SER': count, 'ALERTS': [message]})
                alerts = pd.concat([alerts, new_alert], ignore_index=True)                
            j+=1
        message = f'Total out of limit instances: {count}'
        alerts = pd.concat([alerts, pd.DataFrame({'ALERTS': [message]})], ignore_index=True)
        table_data = alerts.to_dict('records')
        
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'btn_csv_tab4.n_clicks' in changed_id:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, fig, table_data, dcc.send_data_frame(alerts.to_csv, "elog_data.csv"), dash.no_update, dash.no_update
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, fig, table_data, dash.no_update, dash.no_update, dash.no_update

def Slip(selected_mode, lpc_rpm, hpc_rpm, airpr_hpc, ex_temp, n_clicks_submit):
    test_ex = pd.DataFrame(columns = input_cols)
    final_results = pd.DataFrame(columns = input_cols+output_cols)
    fcu_pos = np.round(np.linspace(20, 96, 200), decimals=3)
    test_ex['FCU_POS'] = fcu_pos
    test_ex['INLET_AIR_TEMP'].fillna(40, inplace=True, limit=200)
    final_results['FCU_POS'] = test_ex['FCU_POS']
    final_results['INLET_AIR_TEMP'] = test_ex['INLET_AIR_TEMP']

    case5_pred = loaded_model.predict(test_ex)
    final_results['LPC_RPM'] = case5_pred[:, 0]
    final_results['HPC_RPM'] = case5_pred[:, 1]
    final_results['AIR_PR_AFTER_HPC'] = case5_pred[:, 2]
    final_results['FIELD_TEMP_AVG'] = case5_pred[:, 3]
    final_results['AFTER_HPT_TEMP'] = case5_pred[:, 4]
    final_results['OFF_TAKE_TEMP'] = case5_pred[:, 5]
    final_results['PT_RPM'] = case5_pred[:, 6]
    final_results['S_RPM'] = case5_pred[:, 7]
    final_results[output_cols] = final_results[output_cols].round(3)

    x_data = final_results['HPC_RPM']
    y1 = final_results['LPC_RPM']
    y2 = final_results['AIR_PR_AFTER_HPC']
    y3 = final_results['AFTER_HPT_TEMP']

    # Fit a 2nd degree polynomial and calculate R-squared for each plot
    p1 = np.polyfit(x_data, y1, 2)
    y_pred1 = np.polyval(p1, x_data)
    r_squared1 = 1 - np.sum((y1 - y_pred1)**2) / np.sum((y1 - np.mean(y1))**2)

    p2 = np.polyfit(x_data, y2, 2)
    y_pred2 = np.polyval(p2, x_data)
    r_squared2 = 1 - np.sum((y2 - y_pred2)**2) / np.sum((y2 - np.mean(y2))**2)

    p3 = np.polyfit(x_data, y3, 2)
    y_pred3 = np.polyval(p3, x_data)
    r_squared3 = 1 - np.sum((y3 - y_pred3)**2) / np.sum((y3 - np.mean(y3))**2)

    lp = np.round(np.linspace(lpc_rpm, lpc_rpm, 1))
    hp = np.round(np.linspace(hpc_rpm, hpc_rpm, 1))
    ap = np.round(np.linspace(airpr_hpc, airpr_hpc, 1))
    et = np.round(np.linspace(ex_temp, ex_temp, 1))
    sdf = pd.DataFrame()
    sdf['LPC_RPM'] = lp
    sdf['HPC_RPM'] = hp
    sdf['AIR_PR_AFTER_HPC'] = ap
    sdf['AFTER_HPT_TEMP'] = et

    fig = make_subplots(rows=1, cols=3, shared_xaxes=True, vertical_spacing=0.05)
    # Add predicted lines
    fig.add_trace(go.Scatter(x=x_data, y=y_pred1, name='LPC_RPM', mode='lines', line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_data, y=y_pred2, name='AIR_PR_AFTER_HPC', mode='lines', line=dict(color='blue', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_data, y=y_pred3, name='AFTER_HPT_TEMP', mode='lines', line=dict(color='blue', width=2)), row=1, col=3)

    # Add observed markers
    fig.add_trace(go.Scatter(x=sdf['HPC_RPM'], y=sdf['LPC_RPM'], name='Observed LPC_RPM', mode='markers', marker=dict(color='red', size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sdf['HPC_RPM'], y=sdf['AIR_PR_AFTER_HPC'], name='Observed AIR_PR_AFTER_HPC', mode='markers', marker=dict(color='red', size=8)), row=1, col=2)
    fig.add_trace(go.Scatter(x=sdf['HPC_RPM'], y=sdf['AFTER_HPT_TEMP'], name='Observed AFTER_HPT_TEMP', mode='markers', marker=dict(color='red', size=8)), row=1, col=3)

    # fig.add_trace(row=1, col=1, trace=go.Scatter(x=x_data, y=y1,
    #                     name='LPC_RPM', mode='markers', line=dict(color='red')))
    # fig.add_trace(row=1, col=2, trace=go.Scatter(x=x_data, y=y2,
    #                     name='AIR_PR_AFTER_HPC', mode='markers', line=dict(color='red')))
    # fig.add_trace(row=1, col=3, trace=go.Scatter(x=x_data, y=y3,
    #                     name='AFTER_HPT_TEMP', mode='markers', line=dict(color='red')))
    
    
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
        height=600,
        width=1200,
        title={
            'text': f'<span style="text-decoration: underline;"><b>{selected_mode} - HPC_RPM vs SLIP Parameters</b>',
            'x': 0.5,  # Adjust the title's horizontal position as needed
            'xanchor': 'center',  # Center the title horizontally
            'font': {
                'size': 18,  # Adjust the font size as needed
                'family': 'Arial',  # Specify the font family if needed
            }
        }
    )
    fig.update_yaxes(title_text="LPC_RPM", row=1, col=1)
    fig.update_yaxes(title_text="AIR_PR_AFTER_HPC", row=1, col=2)
    fig.update_yaxes(title_text="AFTER_HPT_TEMP", row=1, col=3)

    fig.add_annotation( text="HPC_RPM",
                        x=0.5, y=-0.1,
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        font=dict(size=18)
                        )
    
    slip_df = pd.DataFrame(columns = input_cols)
    fcu = np.round(np.linspace(58, 58, 1))
    amb_temp = np.round(np.linspace(40, 40, 1))
    slip_df['FCU_POS'] = fcu
    slip_df['INLET_AIR_TEMP'] = amb_temp

    slip_pred = loaded_model.predict(slip_df)
    slip_df['LPC_RPM'] = slip_pred[:, 0]
    slip_df['HPC_RPM'] = slip_pred[:, 1]
    slip_df['AIR_PR_AFTER_HPC'] = slip_pred[:, 2]
    slip_df['AFTER_HPT_TEMP'] = slip_pred[:, 4]

    slip_df = slip_df.applymap(lambda x: round(x,3))
    slip_df[['LPC_RPM', 'HPC_RPM']]=slip_df[['LPC_RPM', 'HPC_RPM']].round()

    slip_lpc = round((slip_df['LPC_RPM'].iloc[0] - lpc_rpm),3)
    slip_air_pr = round((slip_df['AIR_PR_AFTER_HPC'].iloc[0] - airpr_hpc),3)
    slip_ex_temp = round((ex_temp - slip_df['AFTER_HPT_TEMP'].iloc[0]),3)

    slip_alerts = pd.DataFrame(columns=['SLIP Calculation'])

    slip_alerts = pd.concat([slip_alerts, pd.DataFrame({"SLIP Calculation": [f"SLIP (LPC_RPM) (<200 RPM) = Actual(Predicted) LPC RPM - Observed LPC RPM = {slip_df['LPC_RPM'].iloc[0]} - {lpc_rpm} = {slip_lpc}"]})], ignore_index=True)
    slip_alerts = pd.concat([slip_alerts, pd.DataFrame({"SLIP Calculation": [f"SLIP (AIR_PR_AFTER_HPC) (<1 bar) = Actual(Predicted) AIR_PR_AFTER_HPC - Observed AIR_PR_AFTER_HPC = {slip_df['AIR_PR_AFTER_HPC'].iloc[0]} - {airpr_hpc} = {slip_air_pr}"]})], ignore_index=True)
    slip_alerts = pd.concat([slip_alerts, pd.DataFrame({"SLIP Calculation": [f"SLIP (EXHAUST_TEMP) (<30 deg) = Observed EXHAUST_TEMP - Actual(Predicted) EXHAUST_TEMP = {ex_temp} - {slip_df['AFTER_HPT_TEMP'].iloc[0]} = {slip_ex_temp}"]})], ignore_index=True)
    
    # Check if any SLIP value exceeds the threshold
    if (slip_lpc > 200) or (slip_air_pr > 1) or (slip_ex_temp > 30):
        slip_alerts = pd.concat([slip_alerts, pd.DataFrame({'SLIP Calculation': [f"ALERT: Steam or chemical washing required to be undertaken immediately!"]})], ignore_index=True)

    table_data = slip_alerts.to_dict('records')

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, fig, table_data

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=1662, debug=True, use_reloader=False)