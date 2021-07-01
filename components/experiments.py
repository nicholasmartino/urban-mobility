import time
from dash.dependencies import Input, Output
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functions.geopandas import read_gdfs
from Dash_ModeShares import str_gdf
from Predictor import *
from Proximity import Proximity
from SB4_ModeShifts import calculate_mode_shifts


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
        paper_bgcolor='rgba(255,255,255)', plot_bgcolor='rgba(255,255,255)')
)

def experiments_callback(app):
	@app.callback(
		Output('mode_shifts', 'figure'),
		Output('mode_shares', 'figure'),
		Output('emissions', 'figure'),
		[Input('directory', 'value'),
		 Input('sandbox', 'value'),
		 Input('indicator', 'value'),
		 Input('da_baseline', 'value'),
		 Input('baseline_pcl', 'value'),
		 Input('scenarios_pcl', 'value'),
		 Input('baseline_bld', 'value'),
		 Input('scenarios_bld', 'value')],
		#  Input('update_prediction', 'n_clicks')],
		# [State('input-on-submit', 'value')]
	)
	def update_output(folder_path, sandbox, indicator, da_baseline, baseline, scenarios, baseline_b, scenarios_b): #, update_prediction, submit):
			imp_file = f'{directory}/Sandbox/{sandbox} - Features.csv'
			ms_file = f'{directory}/Sandbox/{sandbox} - ModeShifts.csv'
			path = f"{folder_path}/Sandbox/{sandbox}/"

			if not os.path.exists("images"):
				os.mkdir("images")
			initial_path = os.getcwd()

			bsl_prcl = read_gdfs(folder_path, [baseline])
			# bsl_bldg = read_gdfs(folder_path, [baseline_b])
			exp_prcl = read_gdfs(folder_path, scenarios)
			# exp_bldg = read_gdfs(folder_path, scenarios_b)
			base_gdf = gpd.read_feather(f'{folder_path}/{baseline}').to_crs(26910)

			if indicator == 'Mode Shifts':

				# Calculate mode shifts
				ms = calculate_mode_shifts(base_gdf=base_gdf, shares_gdfs=exp_prcl, da_baseline=da_baseline,
										   city_name='Metro Vancouver, British Columbia')

				# Disaggregate modes and experiments data
				all_data = ms.get_all_data().reset_index(drop=True)
				all_data['Sandbox'] = str(sandbox)
				all_data.to_csv(ms_file, index=False)
				df = all_data

				if False:

					print("\n ### Updating predictions ###")

					for scenario, scenario_b in zip(scenarios, scenarios_b):
						predict_mobility(exp_prcl[scenario], exp_bldg[scenario_b], str_gdf, modes, f'{folder_path}')

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
							reg.fitted = pickle.load(open(f'{path}regression/FittedModel_{sandbox}_{rs}.sav', 'rb'))
							reg.train_data = pickle.load(open(f'{path}regression/TrainData_{sandbox}_{rs}.sav', 'rb'))
							reg.feat_imp = pickle.load(open(f'{path}regression/ImportanceData_{sandbox}_{rs}.sav', 'rb'))
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
					ms = calculate_mode_shifts(shares_gdfs={exp.lower(): predictions[exp] for exp in exps}, da_baseline=da_baseline)

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
					if EXPORT:
						imp_stacked.write_image(f'images/{sandbox} - Features.png', scale=15, height=500, width=1200)
						imp_stacked.write_html(f'htmls/{sandbox} - Features.html', scale=15, height=500, width=1200)

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
					if EXPORT:
						imp_hor.write_image(f'images/{sandbox} - Features Horizontal.png', scale=15, height=250, width=1200)
						imp_hor.write_html(f'htmls/{sandbox} - Features Horizontal.html', scale=15, height=250, width=1200)
				else: imp_ver = None

				for mode in modes:
					mode

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
				if EXPORT:
					shares.write_image(f'images/{file_name}', scale=15, height=350, width=1200)
					shares.write_html(f'htmls/{file_name}.html')

				sb_df['Mode Shifts (%)'] = sb_df['∆'] * 100
				shifts = px.box(
					data_frame=sb_df,
					x="Mode",
					y='Mode Shifts (%)',
					facet_col='Experiment',
					color="Mode",
					points=False,
					template=CHART_TEMPLATE,
					color_discrete_map={k.title(): f"#{v}" for k, v in COLOR_MAP.items()},
					title="Mode Shifts"
				)
				shifts.add_shape(  # add a horizontal "target" line
					type="line", line_color="gray", line_width=3, opacity=1, line_dash="dot",
					x0=0, x1=1, xref="paper", y0=0, y1=0, yref="y"
				)
				shifts.update_yaxes(range=[min(sb_df['Mode Shifts (%)'])-2, max(sb_df['Mode Shifts (%)'])+2])
				if EXPORT:
					shifts.write_image(f'images/{sandbox} - Shifts.png', scale=15, height=350, width=1200)
					shifts.write_html(f'htmls/{sandbox} - Shifts.html')

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
					sc_mean = exp_prcl[sc].loc[:, features].mean()
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
				imp_bar.write_html(f'htmls/{sandbox}_importance.html')

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
