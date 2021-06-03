import os
import time

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import geopandas as gpd
import matplotlib
import plotly.express as px
import plotly.graph_objects as go
from Dash_ModeShares import PARCELS, BUILDINGS, STREETS
from Predictor import *
from Proximity import Proximity
from plotly.subplots import make_subplots

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

GEO_PACKAGES = {
    'Sunset': 'Metro Vancouver, British Columbia.gpkg',
    'West Bowl': 'Prince George, British Columbia.gpkg',
    'Hillside Quadra': 'Capital Regional District, British Columbia.gpkg',
}
COLOR_MAP = {"drive": "D62728", "transit": "1F77B4", "walk": "2CA02C", "bike": "FF7F0E"}
EXPORT = True
CHART_TEMPLATE = dict(
    layout=go.Layout(
        title_font=dict(family="Avenir", size=20), font=dict(family="Avenir", size=12),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
)

def get_options(series):
    if series.__class__.__name__ == 'PandasSeries':
        return [{'label': i, 'value': i} for i in series.unique()]

    elif series.__class__.__name__ == 'list':
        return [{'label': i, 'value':i} for i in series]

options = [{'label': i, 'value':i} for i in os.listdir(f"{directory}/Sandbox") if os.path.isdir(f'{directory}/Sandbox/{i}')]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
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
            html.Br(),

            html.Label('DIRECTORY'),
            dcc.Input(id='directory', value=directory, style={'width': '100%'}),

            html.Br(),
            html.Br(),
            html.Label('SANDBOX NAME'),
            dcc.Dropdown(
                id='sandbox',
                options=options,
                value='Main Street'
            ),

            html.Br(),
            html.Label('BASELINE'),
            dcc.Dropdown(
                id='baseline',
                options=get_options(sorted(os.listdir('Regression'))),
            ),

            html.Br(),
            html.Label('SCENARIOS'),
            dcc.Dropdown(
                id='scenarios',
                options=get_options(sorted(os.listdir('Regression'))),
                multi=True
            ),

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

            html.Br(),
            html.Label('UPDATE'),
            daq.ToggleSwitch(id='update_shifts', value=False),

            html.Br(),
            html.Button(['Update'], id='update_prediction'),
            dcc.Input(id='input-on-submit'),
        ]),
    ])
])


@app.callback(
    dash.dependencies.Output('mode_shifts', 'figure'),
    dash.dependencies.Output('mode_shares', 'figure'),
    dash.dependencies.Output('emissions', 'figure'),
    [dash.dependencies.Input('directory', 'value'),
     dash.dependencies.Input('sandbox', 'value'),
     dash.dependencies.Input('indicator', 'value'),
     dash.dependencies.Input('da_baseline', 'value'),
     dash.dependencies.Input('baseline', 'value'),
     dash.dependencies.Input('scenarios', 'value'),
     dash.dependencies.Input('update_shifts', 'value'),
     dash.dependencies.Input('update_prediction', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')]
)
def update_output(folder_path, sandbox, indicator, da_baseline, baseline, scenarios, update_shifts, update_prediction, submit):
        imp_file = f'{directory}/Sandbox/{sandbox} - Features.csv'
        ms_file = f'{directory}/Sandbox/{sandbox} - ModeShifts.csv'
        path = f"{folder_path}/Sandbox/{sandbox}/"

        if not os.path.exists("images"):
            os.mkdir("images")
        initial_path = os.getcwd()

        if indicator == 'Mode Shifts':
            if update_shifts:
                shares_gdfs = {}
                for scenario in scenarios:
                    file_type = scenario.split('.')[len(scenario.split('.')) -1]
                    if file_type == 'feather':
                        gdf = gpd.read_feather(f'Regression/{scenario}').to_crs(26910)
                    else:
                        gdf = gpd.read_file(f'Regression/{scenario}').to_crs(26910)
                    shares_gdfs[scenario] = gdf
                base_gdf = gpd.read_feather(f'Regression/{baseline}').to_crs(26910)

                # Calculate mode shifts
                ms = calculate_mode_shifts(base_gdf=base_gdf, shares_gdfs=shares_gdfs, da_baseline=da_baseline,
                                           city_name='Metro Vancouver, British Columbia')

                # Disaggregate modes and experiments data
                all_data = ms.get_all_data().reset_index(drop=True)
                all_data['Sandbox'] = str(sandbox)
                all_data.to_csv(ms_file, index=False)
                df = all_data

            if update_prediction:
                print("\n ### Updating predictions ###")

                for scenario in scenarios:
                    predict_mobility(PARCELS, BUILDINGS, STREETS, modes, f'Regression/{scenario}')

                if not os.path.exists(f"{path}NetworkAnalysis.sav"):
                    # Update Sandbox analysis
                    start_time = time.time()
                    na = analyze_sandbox({sandbox: experiments[sandbox]}, district=False, export=True)
                    pickle.dump(na, open(f"{path}NetworkAnalysis.sav", 'wb'))
                    print(f"--- {(time.time() - start_time)}s ---")
                else:
                    na = pickle.load(open(f"{path}NetworkAnalysis.sav", 'rb'))

                exps = na[list(na.keys())[0]].keys()

                start_time = time.time()
                predictions = {}
                importance_df = pd.DataFrame()
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
                        reg.feat_imp = pickle.load(open(f'{path}Regression/ImportanceData_{sandbox}_{rs}.sav', 'rb'))
                        predictions[exp][rs] = reg.pre_norm_exp(gdf, export=False)

                        for mode in modes:
                            predictions[exp][rs][f'{mode}_{exp.lower()}_rf_{rs}_n'] = predictions[exp][rs][f'{mode}_rf_n']

                            # Get only essence of feature
                            reg.feat_imp[mode]['feature'] = [i.split('mob_')[1].split('_r')[0] for i in reg.feat_imp[mode]['feature']]
                            reg.feat_imp[mode] = reg.feat_imp[mode].groupby('feature', as_index=False).mean()

                            # Calculate average importance of each feature
                            importance_df = pd.concat([importance_df, reg.feat_imp[mode]['importance']], axis=1)
                            importance_df.columns = list(importance_df.columns)[:-1] + [f'{mode}_{exp.lower()}_{rs}']

                importance_df.index = list(reg.feat_imp[mode]['feature'])
                importance_df = importance_df.drop(['drive_length', 'network_drive_ct', 'network_bike_ct'])

                # Disaggregate feature importance
                importance_dis = pd.DataFrame()
                for feature in importance_df.index:
                    for mode in modes:
                        i = len(importance_dis)
                        importance_dis.loc[i, ['feature', 'mode']] = [feature, mode.title()]
                        importance_dis.at[i, 'importance'] = importance_df.loc[feature, [col for col in importance_df.columns if mode in col]].mean()
                        importance_dis.at[i, 'total'] = importance_df.loc[feature, [col for col in importance_df.columns]].mean()
                importance_dis = importance_dis.sort_values(['total'], ascending=False)
                importance_dis.to_csv(f'{directory}Sandbox/{sandbox} - Features.csv', index=False)

                # Calculate mode shifts
                ms = calculate_mode_shifts(
                    sandbox=sandbox, shares_gdfs={exp.lower(): predictions[exp] for exp in exps}, da_baseline=da_baseline)

                # Get trip demand for region
                demand = False
                file_path = f"{'/'.join(os.path.realpath(__file__).split('/')[:-1])}/Maps"
                print(file_path)
                if demand or not os.path.exists(f'{file_path}/{sandbox} - Blocks.feather'):
                    gpk = GEO_PACKAGES[sandbox]
                    destinations = gpd.read_file(f'{directory}/{gpk}', layer='land_assessment_parcels')
                    blocks = estimate_demand(ms.block, destinations)
                    blocks.to_feather(f"{file_path}/{sandbox} - Blocks.feather")
                else:
                    blocks = gpd.read_feather(f'{file_path}/{sandbox} - Blocks.feather')

                # Estimate emissions
                parcels = ms.block.copy()
                for exp in exps:
                    em = calculate_emissions(ms.block, blocks, f"_{exp}".lower())
                    parcels = pd.concat([parcels, em.loc[:, [col for col in em.columns if
                        ('drive_em' in col) or ('transit_em' in col) or ('walkers' in col) or
                        ('bikers' in col) or ('riders' in col) or ('drivers' in col)]]], axis=1)
                ms.block = parcels

                # Disaggregate modes and experiments data
                all_data = ms.get_all_data().reset_index(drop=True)
                all_data['Sandbox'] = sandbox
                all_data.to_csv(ms_file, index=False)
                df = all_data
                print(f"--- {(time.time() - start_time)}s --- {directory}Sandbox/{sandbox} - ModeShifts.csv")
            # else:
                # importance_dis = pd.read_csv(imp_file)
                # df = pd.read_csv(ms_file)
                # exps = list(df['Experiment'].unique())

            importance=False
            if importance:
                mapper = {
                    'frequency': 'Pub. transit freq.',
                    'population density per square kilometre, 2016': 'Population dens.',
                    'number_of_bedrooms': 'No. bedrooms',
                    'n_dwellings': 'No. dwellings',
                    'gross_building_area': 'Gross bldg. area',
                    'walk_length': 'Walk network lg.',
                    'total_finished_area': 'Finished area',
                    'population, 2016': 'Population',
                    'drive_length': 'Drive network lg.',
                    'network_walk_ct': 'No. st. segments',
                    'bike_length': 'Bike network lg.',
                    'land_assessment_fabric_ct': 'No. parcels',
                    'CM': 'No. comm. prcls.',
                    'MX': 'No. mixed prcls.',
                    'MFL': 'No. multi-family prcls.',
                    'SFA': 'No. 1-family attached prcls.',
                    'SFD': 'No. 1-family detached prcls.',
                    'network_stops_ct': 'No. transit stops',
                    'network_drive_ct': 'No. driveable segments',
                    'network_bike_ct': 'No. bikeable segments'
                }
                importance_dis['feature'] = importance_dis['feature'].replace(mapper)

                # Importance stacked vertical bar chart
                imp_stacked = px.bar(
                    importance_dis, x="feature", y="importance", color="mode",
                    color_discrete_map={k.title(): f"#{v}" for k, v in COLOR_MAP.items()}, template=CHART_TEMPLATE)
                imp_stacked.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                if EXPORT: imp_stacked.write_image(f'images/{sandbox} - Features.png', scale=15, height=500, width=1200)

                # Importance simple horizontal bar chart
                imp_hor = make_subplots(rows=1, cols=len(modes), horizontal_spacing=0.12)
                imp_ver = make_subplots(rows=len(modes), cols=1)
                for i, mode in enumerate(modes):
                    mode_df = importance_dis[importance_dis['mode'] == mode.title()]
                    mode_df = mode_df.sort_values('importance').tail(5)
                    trace = go.Bar(
                        x=list(mode_df['importance']), y=list(mode_df['feature']),
                        orientation='h', name=mode.title(), marker_color=f"#{COLOR_MAP[mode]}", showlegend=False,)
                    imp_hor.add_trace(trace, row=1, col=i+1)
                    imp_ver.add_trace(trace, row=i+1, col=1)
                imp_hor.update_xaxes(title="Permutation Imp.")
                imp_hor.update_layout(template=CHART_TEMPLATE, margin=dict(t=40,b=50,l=120,r=0), title="Permutation Importance", font = dict(family="Avenir", size=16))
                imp_ver.update_layout(template=CHART_TEMPLATE, margin=dict(l=120), title="Permutation Importance")
                if EXPORT: imp_hor.write_image(f'images/{sandbox} - Features Horizontal.png', scale=15, height=250, width=1200)
            else: imp_ver = None

            # Display mode shares on box plot
            os.chdir(initial_path)
            sb_df = df[df['Sandbox'] == str(sandbox)].dropna(how='all')
            sb_df = sb_df.sort_values(['Order'], ascending=True)
            shares = make_subplots(rows=1, cols=len(scenarios)+1, specs=[[{'type': 'domain'} for j in range(len(scenarios)+1)]], row_titles=['Mode Shares'])
            for i, exp in enumerate(['e0'] + scenarios):
                values = [sb_df[(sb_df["Mode"] == mode.title()) & (sb_df["Experiment"] == exp)]['Share'].mean() for mode in modes]
                shares.add_trace(go.Pie(sort=False,
                    labels=[mode.title() for mode in modes],
                    values=values,
                    name=f"{exp.title()} Mode Shares",
                ), row=1, col=i + 1)
                print(f'{exp} - {values}')
            shares.update_traces(hole=0.618, hoverinfo="label+percent+name")
            shares.update_layout(margin=dict(t=0,b=0,l=60,r=0), template=CHART_TEMPLATE)
            if da_baseline: file_name = f'{sandbox} - Shares (DA).png'
            else: file_name = f'{sandbox} - Shares (Predicted).png'
            if EXPORT: shares.write_image(f'images/{file_name}', scale=15, height=350, width=1200)

            sb_df['∆'] = sb_df['∆'] * 100
            shifts = px.box(
                data_frame=sb_df,
                x="Mode",
                y='∆',
                facet_col='Experiment',
                color="Mode",
                points=False,
                template=CHART_TEMPLATE,
                color_discrete_map={k.title(): f"#{v}" for k, v in COLOR_MAP.items()},
                title="Mode Shifts"
            )
            shifts.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            shifts.add_shape(  # add a horizontal "target" line
                type="line", line_color="gray", line_width=3, opacity=1, line_dash="dot",
                x0=0, x1=1, xref="paper", y0=0, y1=0, yref="y"
            )
            shifts.update_yaxes(range=[min(sb_df['∆'])-2, max(sb_df['∆'])+2])
            if EXPORT: shifts.write_image(f'images/{sandbox} - Shifts.png', scale=15, height=350, width=1200)

            ### Create line chart with changes in most important features
            imp = pd.DataFrame()
            for sc in scenarios:
                imp_sc = pd.read_csv(f'tables/importance_{sc}.csv')
                imp_sc['Experiment'] = sc
                imp = pd.concat([imp, imp_sc])

            # Get most important features
            features = imp.groupby('feature').sum().sort_values('importance', ascending=False).index[:10]
            imp_feat = pd.DataFrame()
            for i, sc in enumerate(scenarios):
                sc_mean = shares_gdfs[sc].loc[:, features].mean()
                base_mean = base_gdf.loc[:, features].mean()
                sc_df = pd.DataFrame(sc_mean, columns=[f'Value'])
                sc_df[f'Change (%)'] = ((sc_mean - base_mean)/base_mean) * 100
                sc_df['Experiment'] = i + 1
                imp_feat = pd.concat([imp_feat, sc_df])

            imp_feat['Feature'] = imp_feat.index
            imp_feat = imp_feat.reset_index(drop=True).sort_values('Experiment').replace({
                'mob_frequency_r1200_ave_f': 'μ Transit Freq. 1200',
                'mob_frequency_r1200_sum_f': '∑ Transit Freq. 1200',
                'mob_CM_r1200_sum_f': '∑ Commercial 1200',
                'mob_CM_r1200_ave_f': 'μ Commercial 1200',
                'mob_population density per square kilometre, 2016_r1200_sum_f': '∑ Pop. Density 1200',
                'mob_n_dwellings_r1200_sum_f': '∑ No. Dwellings 1200',
                'mob_population density per square kilometre, 2016_r1200_ave_f': 'μ Pop. Density 1200',
                'mob_frequency_r800_sum_f': '∑ Transit Freq. 800',
                'mob_n_dwellings_r400_ave_f': 'μ No. Dwellings 400',
                'mob_n_dwellings_r1200_ave_f': 'μ No. Dwellings 1200'
            })
            imp_bar = px.bar(imp_feat, x='Experiment', y='Change (%)', facet_col='Feature', color='Feature', barmode="group", template=CHART_TEMPLATE)
            imp_bar.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            imp_bar.update_layout(showlegend=False)

            emissions=False
            if emissions:
                # Display emissions
                em_df = df[df['Sandbox'] == sandbox].dropna(how='all').copy(deep=False)
                for i, mode in enumerate(modes):
                    em_df.loc[em_df["Mode"] == mode.title(), "Order"] = i
                em_df = em_df[(em_df['Sandbox'] == sandbox)].groupby(['Experiment', 'Mode'], as_index=False).sum()
                em_df['Emissions (tCO2/yr.)'] = (em_df['Emissions'] * 2 * 260)/1000000
                em_df = em_df.sort_values(["Order"], ascending=True)
                y = 'Emissions (tCO2/yr.)'
                margin = 0.1
                em = px.line(em_df, x='Experiment', y=y, range_y=[min(em_df[em_df[y] > 0][y]) * (1 - margin), max(em_df[em_df[y] > 0][y]) * (1 + margin)],
                             line_group="Mode", color="Mode", template=CHART_TEMPLATE)
                em.update_traces(mode='markers+lines')
            else: em = imp_bar

            return shifts, shares, em

        elif indicator == 'Proximity':

            bl = Proximity(baseline)
            for exp in experiments:
                prx = Proximity(exp)


            return [prx_maps, None, None]

if __name__ == '__main__':
    app.run_server(debug=False, host='localhost', port=9050)
