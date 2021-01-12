import pickle
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from SB0_Variables import *
from plotly.subplots import make_subplots

start_time = time.time()
from GeoLearning.Supervised import Regression
print(f"--- {(time.time() - start_time)}s ---")
start_time = time.time()
from UrbanMobility.SB3_AnalyzeSandbox import analyze_sandbox
print(f"--- {(time.time() - start_time)}s ---")
start_time = time.time()
from UrbanMobility.SB4_ModeShifts import calculate_mode_shifts
print(f"--- {(time.time() - start_time)}s ---")
print("Import finished")

df = pd.read_csv(f'{directory}Sandbox/ModeShifts.csv')

def update_output(folder_path, sandbox, y, update_prediction, submit):
    path = f"{folder_path}Sandbox/{sandbox}/"
    sb_df = df[df['Sandbox'] == sandbox]
    exps = sb_df['Experiment'].unique()

    update_na = False
    if update_na:
        # Update Sandbox analysis
        start_time = time.time()
        na = analyze_sandbox({sandbox: experiments[sandbox]}, district=False, export=False)
        pickle.dump(na, open(f"{path}NetworkAnalysis.sav", 'wb'))
        print(f"--- {(time.time() - start_time)}s ---")

    if update_prediction:
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

            predictions[exp] = pd.concat([i for rs, i in predictions[exp].items()]).groupby(level=0).mean()

        # Calculate mode shifts
        ms = calculate_mode_shifts(sandbox=sandbox, shares_gdfs={exp.lower(): predictions[exp] for exp in exps})
        all_data = ms.get_all_data().reset_index(drop=True)
        all_data['Sandbox'] = sandbox
        df.loc[(df['Sandbox'] == sandbox), :] = all_data
        df.fillna(method='ffill', inplace=True)
        df.to_csv(f'{directory}Sandbox/ModeShifts.csv', index=False)
        print(f"--- {(time.time() - start_time)}s ---")

    # Display mode shares
    shares = make_subplots(rows=1, cols=len(exps), specs=[[{'type': 'domain'} for j in range(len(exps))]])
    for i, exp in enumerate(exps):
        shares.add_trace(go.Pie(
            labels=["Active", "Transit", "Drive"],
            values=[df[df["Mode"] == mode]['Share'].sum()/df["Share"].sum() for mode in modes],
            hole=0.38, name=f"{exp.title()} Mode Shares"), row=1, col=i + 1)

    # Display mode shifts
    shifts = px.box(
        data_frame=df,
        x="Mode",
        y=y,
        facet_col='Experiment',
        points='all'
    )
    return shares, shifts

update_output(directory, 'West Bowl', '∆', True, None)
