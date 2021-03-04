import dash
import pandas as pd
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import pickle
import time
import plotly.graph_objects as go
from SB0_Variables import *
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import matplotlib
import os
import gc
import geopandas as gpd
import dash_daq as daq
matplotlib.use('Agg')

start_time = time.time()
from GeoLearning.Supervised import Regression
print(f"--- {(time.time() - start_time)}s ---")
start_time = time.time()
from UrbanMobility.SB3_AnalyzeSandbox import analyze_sandbox
print(f"--- {(time.time() - start_time)}s ---")
start_time = time.time()
from UrbanMobility.SB4_ModeShifts import calculate_mode_shifts
print(f"--- {(time.time() - start_time)}s ---")
from Sandbox import calculate_emissions, estimate_demand
print("Import finished")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

geo_packages = {
    'Sunset': 'Metro Vancouver, British Columbia.gpkg',
    'Prince George': 'Prince George, British Columbia.gpkg',
    'Hillside Quadra': 'Capital Regional District, British Columbia.gpkg',
}

df = pd.read_csv(f'{directory}Sandbox/ModeShifts.csv')
df['id'] = df.index
print(f"Loaded DataFrame with {len(df)} rows")

def generate_options(series):
    return [{'label': i, 'value': i} for i in series.unique()]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(children=[
    dbc.Row([
        dbc.Col(style={'width': '13.5%', 'display':'inline-block'}, children=[
            html.Label('Directory'),
            dcc.Input(id='directory', value=directory),
            html.Label('Sandbox'),
            dcc.Dropdown(
                id='sandbox',
                options=generate_options(df['Sandbox']),
                value='Sunset'
            ),
            html.Label('Y-Axis'),
            dcc.Dropdown(
                id='y',
                options=generate_options(df.columns),
                value='∆'
            ),
            html.Br(),

            html.Button(['Update'], id='update_prediction'),
            dcc.Input(id='input-on-submit'),

            html.Br(),
            html.Label('DA Baseline'),
            daq.ToggleSwitch(id='da_baseline', value=True)

        ]),
        dbc.Col(style={'width': '85%', 'float': 'right', 'display':'inline-block'}, children=[
            dcc.Graph(id='mode_shifts'),
            dcc.Graph(id='mode_shares'),
        ])
    ])
])


@app.callback(
    dash.dependencies.Output('mode_shares', 'figure'),
    dash.dependencies.Output('mode_shifts', 'figure'),
    [dash.dependencies.Input('directory', 'value'),
     dash.dependencies.Input('sandbox', 'value'),
     dash.dependencies.Input('y', 'value'),
     dash.dependencies.Input('da_baseline', 'value'),
     dash.dependencies.Input('update_prediction', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')]
)
def update_output(folder_path, sandbox, y, da_baseline, update_prediction, submit):
        path = f"{folder_path}Sandbox/{sandbox}/"
        sb_df = df[df['Sandbox'] == sandbox]
        exps = sb_df['Experiment'].unique()

        if not os.path.exists("images"):
            os.mkdir("images")
        initial_path = os.getcwd()

        if update_prediction:

            print("\n ### Updating predictions ###")
            update_na = False
            if update_na:
                # Update Sandbox analysis
                start_time = time.time()
                na = analyze_sandbox({sandbox: experiments[sandbox]}, district=False, export=True)
                pickle.dump(na, open(f"{path}NetworkAnalysis.sav", 'wb'))
                print(f"--- {(time.time() - start_time)}s ---")

            start_time = time.time()
            na = pickle.load(open(f"{path}NetworkAnalysis.sav", 'rb'))
            predictions = {}
            for exp in exps:

                # Read Sandbox GeoDataFrame
                gdf = na[f"{sandbox}"][f"{exp.lower()}"]
                reg = Regression(
                    directory=f'{path}',
                    predicted=modes,
                    test_split=0.2,
                    round_f=4,
                    norm_x=False,
                    norm_y=False,
                    prefix=f'',
                    filter_pv=True,
                    plot=False,
                    pv=0.05,
                )
                predictions[exp] = {}
                for rs in range(r_seeds):

                    # Predict mode shares
                    predictions[exp][rs] = {}
                    reg.fitted = pickle.load(open(f'{path}Regression/FittedModel_{sandbox}_{rs}.sav', 'rb'))
                    reg.train_data = pickle.load(open(f'{path}Regression/TrainData_{sandbox}_{rs}.sav', 'rb'))
                    predictions[exp][rs] = reg.pre_norm_exp(gdf, export=False)
                    for mode in modes:
                        predictions[exp][rs][f'{mode}_{exp.lower()}_rf_{rs}_n'] = predictions[exp][rs][f'{mode}_rf_n']

            # Calculate mode shifts
            ms = calculate_mode_shifts(sandbox=sandbox, shares_gdfs={exp.lower(): predictions[exp] for exp in exps}, da_baseline=da_baseline)

            # Get trip demand for region
            gpk = geo_packages[sandbox]
            destinations = gpd.read_file(f'{directory}/{gpk}', layer='land_assessment_parcels')
            blocks = estimate_demand(ms.block, destinations)

            # Estimate emissions
            new_blocks = gpd.GeoDataFrame()
            for exp in exps:
                em = calculate_emissions(ms.block, blocks, f"_{exp}".lower())
                new_blocks = pd.concat([new_blocks, em.loc[:, [col for col in em.columns if 'total_em_kg_yr_e' in col]].mean(axis=1)], axis=1)
            new_blocks.columns = [f"{e}_emissions" for e in exps]
            ms.block = pd.concat([ms.block, new_blocks], axis=1)

            all_data = ms.get_all_data().reset_index(drop=True)
            all_data['Sandbox'] = sandbox
            all_data.index = df.loc[(df['Sandbox'] == sandbox), :].index
            df.loc[(df['Sandbox'] == sandbox), :] = all_data

            df.sort_values("id")
            df.to_csv(f'{directory}Sandbox/ModeShifts.csv', index=False)
            print(f"--- {(time.time() - start_time)}s ---")

        # Display emissions
        em = px.line(df, x='Experiment', y='Emissions')
        em.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        # Display mode shares
        os.chdir(initial_path)
        sb_df = df[df['Sandbox'] == sandbox].dropna(how='all')
        sb_df = sb_df.sort_values(['Experiment', 'Mode'], ascending=True)
        shares = make_subplots(rows=1, cols=len(exps), specs=[[{'type': 'domain'} for j in range(len(exps))]])
        for i, exp in enumerate(exps):
            shares.add_trace(go.Pie(sort=False,
                labels=[mode.title() for mode in modes],
                values=[sb_df[(sb_df["Mode"] == mode.title()) & (sb_df["Experiment"] == exp)]['Share'].mean() for mode in modes],
                name=f"{exp.title()} Mode Shares",
            ), row=1, col=i + 1)

            for mode in modes:
                print(f"{sandbox} ({exp}) {mode.title()} - {round(100 * (sb_df[(sb_df['Mode'] == mode.title()) & (sb_df['Experiment'] == exp)]['Share'].mean() - sb_df[(sb_df['Mode'] == mode.title()) & (sb_df['Experiment'] == 'E0')]['Share'].mean()), 2)}")

        shares.update_traces(hole=0.618, hoverinfo="label+percent+name")
        shares.update_layout(margin=dict(t=0,b=0,l=60,r=0), height=270, template="simple_white",
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        if da_baseline: file_name = f'{sandbox} - Shares (DA).png'
        else: file_name = f'{sandbox} - Shares (Predicted).png'
        shares.write_image(f'images/{file_name}', scale=15, height=350, width=1200)

        # Display mode shifts
        for i, mode in enumerate(modes):
            sb_df.loc[sb_df["Mode"] == mode.title(), "Order"] = i
        sb_df = sb_df.sort_values(["Order", "Experiment"], ascending=True)

        shifts = px.box(
            data_frame=sb_df,
            x="Mode",
            y=y,
            facet_col='Experiment',
            points='all',
            color="Mode",
            template="simple_white",
        )
        shifts.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        shifts.add_shape(  # add a horizontal "target" line
            type="line", line_color="gray", line_width=3, opacity=1, line_dash="dot",
            x0=0, x1=1, xref="paper", y0=0, y1=0, yref="y"
        )
        shifts.update_yaxes(range=[-0.3, 0.7])
        shifts.write_image(f'images/{sandbox} - Shifts.png', scale=15, height=350, width=1200)
        return shares, shifts


if __name__ == '__main__':
    app.run_server(debug=False, host='localhost')
