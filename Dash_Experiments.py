import sys
import time

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import matplotlib
import plotly.graph_objects as go
from dash import Dash

from components.experiments import experiments_callback
from components.settings import settings_layout, settings_callback
from functions.dash import get_options

matplotlib.use('Agg')

start_time = time.time()
sys.path.insert(1, "/Volumes/Macintosh HD/Users/nicholasmartino/Google Drive/Python/GeoLearning")
print(f"--- {(time.time() - start_time)}s ---")
start_time = time.time()
print(f"--- {(time.time() - start_time)}s ---")
start_time = time.time()
print(f"--- {(time.time() - start_time)}s ---")
print("Import finished")

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__)# , external_stylesheets=external_stylesheets)
app.layout = html.Div(children=[
    dbc.Row([

        dbc.Col(style={'width': '80%', 'display':'inline-block'}, children=[
            dcc.Graph(id='mode_shifts', style={'height': '50vh'}),
            dcc.Graph(id='mode_shares', style={'height': '20vh'}),
            dcc.Graph(id='emissions', style={'height': '30vh'}),
        ]),

        dbc.Col(style={'width': '20%', 'float': 'right', 'display':'inline-block', 'margin-left': '2em', 'margin-right': '2em'}, children=[

            html.Br(),
            html.Header('PARAMETERS', style={'text-align': 'center', 'width': '100%', 'font-weight': 'bold',
                                             'font-size':'medium', 'display':'inline-block'}),
            html.Br(),
            html.Label(
              'SANDBOX NAME'),
            dcc.Input(
              id='sandbox',
              value='Main Street'
            ),
            html.Br()]+settings_layout()+[

            html.Br(),
            html.Br(),

            html.Br(),
            html.Label('INDICATOR'),
            dcc.Dropdown(
                id='indicator',
                options=get_options(['Mode Shifts', 'Proximity']),
                value='Mode Shifts'
            ),

            html.Br(),
            html.Label('CENSUS BASELINE'),
            daq.ToggleSwitch(id='da_baseline', value=True),

        ]),
    ])
])

settings_callback(app)
experiments_callback(app)

if __name__ == '__main__':
    app.run_server(debug=False, host='localhost', port=9050)
