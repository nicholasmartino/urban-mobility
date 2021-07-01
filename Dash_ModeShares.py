# -*- coding: utf-8 -*-
import dash
import pandas as pd
import geopandas as gpd
import dash_daq as daq
import plotly.express as px
import json
import os
from functions.dash import get_options
import gc
import dash_table
from functions.geopandas import read_gdfs, export_multi
import dash_deck
import pydeck as pdk
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from Predictor import *
from SB0_Variables import *
from matplotlib import colors, cm
from shapely.geometry import Polygon
from Visualization.Colors import get_rgb_values
from UrbanZoning.Layers import Layers
from UrbanZoning.City.Fabric import Neighbourhood
import pickle
import matplotlib
matplotlib.use('Agg')

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

exp = "E3"
# BUILDINGS = f"sandbox_bldgs_{exp}.shp"
# PARCELS = f"sandbox_prcls_{exp}.shp"
# STREETS = f"sandbox_strts_{exp}.shp"

api_keys = {'mapbox': "pk.eyJ1IjoibmljaG9sYXNtYXJ0aW5vIiwiYSI6ImNrMjVhOGphOTAzZGUzbG8wNHJhdTZrMmYifQ.98uDMnGIvn1zrw4ZWUO35g"}

sb_name = 'Main Street'
pdk_layers = {}

def filter_buffer(gdf, layer):
	gdf = gdf[gdf[layer] == 1]
	gdf['geometry'] = gdf.buffer(5)
	return gdf.to_crs(4326)

str_gdf = gpd.read_file(f"/Volumes/Macintosh HD/Users/nicholasmartino/Google Drive/elementslab/main_st_streets_e0.geojson")

app.layout = \
	dbc.Col(
		style={'width': '100%', 'offset': 0, 'display': 'inline-block'},
		children=[
			dcc.Store(id='map_mode'),
			dcc.Store(id='memory'),
			html.Div(id="deck_div", style={'.bs-tooltip-left': {'top': '-12px', 'left': '-182px'}}),

			dbc.Col(style={'width': '300px', 'display': 'inline-block'}, className='pretty_container', children=[
				html.I("Define File Paths"),
				html.Br(),
				html.Br(),
				html.Label("DIRECTORY"),
				dcc.Input(id="direct", type="text", style={'width': '100%'},
				          value=DIRECTORY),

				html.Br(),
				html.Br(),
				html.Label("PARCELS"),
				html.Div(id='parcels'),

				html.Br(),
				html.Label("BUILDINGS"),
				html.Div(id='buildings'),

				html.Br(),
				html.Label("SAMPLES"),
				html.Div(id='samples'),

				# html.Br(),
				# html.Label("Streets"),
				# dcc.Input(id="streets", type="text", style={'width': '100%'},
				# 		  value=STREETS),

				html.Br(),
				html.Label("MODES"),
				dcc.Dropdown(id='color_by', style={'width': '100%'},
				             options=[{'label': i.title(), 'value': i} for i in MOB_MODES], value=['walk'], multi=True),

				html.Br(),
				html.Label("EXPORT"),
				dbc.Col(style={'width': '100%'}, children=[
					dbc.Row([dcc.Dropdown(id='export_format', style={'width': '80%'}, multi=True,
					                      options=get_options(['.feather', '.geojson', '.shp']), value=['.feather']),
					         daq.ToggleSwitch(id='export', style={'width': '50%'}, value=True)]),
				]),
				html.Br(),
				html.Label("Run"),
				daq.ToggleSwitch(id='run', value=False),

				html.Br(),
				html.Br(),
				# dash_table.DataTable(
				# 	id='attribute_table',
				# 	columns=([{"name": 'Feature', "id": 'Feature'}, {"name": 'Mean', "id": 'Mean'}, {"name": 'Max', "id": 'Max'}]),
				# 	data=[],
				# 	style_as_list_view=True,
				# ),
				dcc.Store(id="click-info-json-output"),
				html.Pre(id="click-event-json-output"),
			]),

			# dbc.Col(style={'width': '50%', 'display': 'inline-block'}, children=[
			#     dcc.Graph(id="map_view"),
			# ]),

			dbc.Col(style={'width': '20%', 'display': 'inline-block', 'float': 'right'}, className='pretty_container', children=[
				dbc.Row([
					dcc.Graph(id="shares", style={'width': '80%', 'height': '45vh'}),
				]),
				dbc.Row([
					dcc.Graph(id="hist", style={'width': '100%', 'height': '45vh'}),
				]),
			])
		])


@app.callback(Output('buildings', 'children'),
              Output('parcels', 'children'),
              Output('samples', 'children'),
              [Input('direct', 'value')])
def update_menu(direct):
	buildings = dcc.Dropdown(id="buildings", style={'width': '100%'}, options=get_options(sorted(os.listdir(direct))), multi=True),
	parcels = dcc.Dropdown(id="parcels", style={'width': '100%'}, options=get_options(sorted(os.listdir(direct))), multi=True),
	samples = dcc.Dropdown(id='samples', style={'width': '100%'}, options=get_options(sorted(os.listdir(direct))), multi=True)
	return buildings, parcels, samples


# def assign_callback(app, out, event):
# 	@app.callback(Output(f"attribute_table", "data"), [Input("deck_div", event)])
# 	def dump_json(data):
#
# 		try:
# 			data['object']['geometry'] = [Polygon(coord) for coord in data['object']['geometry']['coordinates']]
# 			data['object'] = {key: data['object'][key] for key in data['object'].keys() & {'walk', 'bike', 'transit', 'drive'}}
# 			df = pd.DataFrame(data['object'], index=[0])
#
# 			return dtb.DataTable(df.to_dict('records'))
#
# 		except: pass
#
# assign_callback(app, "click-info", "clickInfo")


@app.callback(
	Output("deck_div", "children"),
	Output("shares", "figure"),
	Output("hist", "figure"),
	# Output("attribute_table", "data"),
	Input("direct", "value"),
	Input("parcels", "value"),
	Input("buildings", "value"),
	Input("samples", "value"),
	Input("color_by", "value"),
	Input("run", "value"),
	Input("export", "value"),
	Input("export_format", "value"),
	Input("click-info-json-output", "data")
)
def update_output(direct, parcels, buildings, samples, color_by, run, export, export_formats, memory):

	print(memory)
	gdfs = read_gdfs(direct, samples)

	if len(samples) != 1:
		assert len(color_by) == len(samples), AssertionError("Number of samples different than number of modes")
	assert len(parcels) == len(buildings), AssertionError("Number of parcel layers does not equal number of buildings layer")

	# tooltip = {
	# 	"html": f"{color_by.title()} "+"{"f"{color_by}"+"}"
	# }

	r = pdk.Deck(
		layers=[],
		initial_view_state=pdk.ViewState(latitude=0, longitude=0, zoom=15, max_zoom=16, pitch=0, bearing=0),
		api_keys=api_keys,
		map_style=pdk.map_styles.LIGHT,
	)
	lyr = Layers()
	all_predicted = pd.DataFrame()

	for parcel, building in zip(parcels, buildings):

		if run:
			samples_analyzed = []
			for sample in samples:
				proxy_gdf = analyze_sandbox(
					buildings=gpd.read_file(f"{direct}/{building}"),
					parcels=gpd.read_file(f"{direct}/{parcel}"),
					streets=str_gdf,
					sample_gdf=gdfs[sample],
					suffix=parcel,
				    ch_dir=direct)
				samples_analyzed.append(proxy_gdf)
		else:
			samples_analyzed = gpd.read_feather(f'{direct}/Network/Sandbox_mob_{parcel}_na.feather')

		if len(samples) == 1:
			samples_analyzed = [s for i in color_by for s in samples_analyzed]
			samples = [s for i in color_by for s in samples]

		for mode, sample_gdf, sample in zip(color_by, samples_analyzed, samples):
			if run:
				predicted = predict_mobility(gdf=sample_gdf, mob_modes=[mode])
				predicted['Sample'] = sample
				predicted['Parcels'] = parcel
				predicted['Mode'] = mode
				export_multi(predicted, export_formats, directory=f'{direct}/Mode Shares', file=f'ModeShares_{sample}_{parcel}_{mode}')
			else:
				predicted = gpd.read_feather(f'{direct}/Mode Shares/ModeShares_{sample}_{parcel}_{mode}.feather')

			build_choropleth_map(predicted, mode, lyr)
			all_predicted = pd.concat([all_predicted, predicted])

	predicted = all_predicted
	shares = build_pie_chart(predicted)
	hist = build_histogram(predicted)
	predicted.to_file(f'{direct}/Mode Shares/Mode Shares.feather')

	if export:
		shares.write_image(f'images/{sb_name} - {parcels} - Mode Shares.png')
		hist.write_image(f'images/{sb_name} - {parcels} - Mode Shares - Histogram.png')

	# def load_importance_data():
	# 	# Load importance data
	# 	importance = load_importance()
	# 	if export: importance.to_csv(f'tables/importance_{sb_name}_{parcels}.feather.csv')
	# 	importance = importance.groupby(['key', 'feature']).sum()
	# 	importance  = importance.loc[color_by, :].sort_values('importance', ascending=False)
	# 	prd = predicted.loc[:, list(importance.index)[:6]]
	# 	att_table = pd.concat([pd.DataFrame(prd.mean()), pd.DataFrame(prd.max())], axis=1).round(2).reset_index()
	# 	att_table.columns = ['Feature', 'Mean', 'Max']
	# 	return att_table

	# # Export individual histograms
	# for mode in MOB_MODES:
	# 	hist2 = px.histogram(predicted, x=mode, color_discrete_sequence=[f"#{COLOR_MAP[mode]}"])
	# 	hist2.update_layout(margin={'b': 0, 'l': 0, 'r': 0, 't': 0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
	# 	if export: hist2.write_image(f'images/{sb_name} - {parcels} - Mode Shares - Histogram ({mode.title()}).png')

	if color_by is not None:
		r.layers = list(r.layers) + [lyr.layers[key][0] for key in lyr.layers.keys() if len(lyr.layers[key]) > 0]
		# print(f"{color_by} - {[l.id for l in r.layers]} - {MODES_COLORS[color_by].name}")

	print([layer.id for layer in r.layers])

	# Get centroid
	centroid = all_predicted.unary_union.centroid
	r.initial_view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=15, max_zoom=16, pitch=0, bearing=0)
	r.update()

	dgl = dash_deck.DeckGL(r.to_json(), id="deck", mapboxKey=r.mapbox_key, enableEvents=['click'], # tooltip=tooltip,
	                       style={'width': '100%', 'float': 'left', 'display': 'inline-block'})
	gc.collect()
	# if export: att_table.to_csv(f'tables/{sb_name}_{parcels}_Important_{color_by.title()}.csv')
	return [dgl, shares, hist]


if __name__ == "__main__":
	app.run_server(debug=False, host='localhost', port=7050)
