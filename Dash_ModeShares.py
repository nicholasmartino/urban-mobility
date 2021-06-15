# -*- coding: utf-8 -*-
import dash
import pandas as pd
import geopandas as gpd
import dash_daq as daq
import plotly.express as px
import json
import gc
import dash_table
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
DIRECTORY = "/Volumes/ELabs/50_projects/20_City_o_Vancouver/SSHRC Partnership Engage/Sandbox/shp/MainSt/Experiment"
BUILDINGS = f"sandbox_bldgs_{exp}.shp"
PARCELS = f"sandbox_prcls_{exp}.shp"
STREETS = f"sandbox_strts_{exp}.shp"

api_keys = {'mapbox': "pk.eyJ1IjoibmljaG9sYXNtYXJ0aW5vIiwiYSI6ImNrMjVhOGphOTAzZGUzbG8wNHJhdTZrMmYifQ.98uDMnGIvn1zrw4ZWUO35g"}
modes_colors = {'transit': cm.Blues, 'drive': cm.Reds, 'walk': cm.Greens, 'bike': cm.Oranges}
mob_modes = list(modes_colors.keys())

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
				html.Label("Directory"),
				dcc.Input(id="direct", type="text", style={'width': '100%'},
				          value=DIRECTORY),

				html.Br(),
				html.Label("Parcels"),
				dcc.Input(id="parcels", type="text", style={'width': '100%'},
						  value=PARCELS),
				html.Br(),
				html.Label("Buildings"),
				dcc.Input(id="buildings", type="text", style={'width': '100%'},
						  value=BUILDINGS),
				# html.Br(),
				# html.Label("Streets"),
				# dcc.Input(id="streets", type="text", style={'width': '100%'},
				# 		  value=STREETS),

				html.Br(),
				html.Br(),
				html.Label("Export"),
				daq.ToggleSwitch(id='export', value=True),
				html.Br(),
				html.Label("Run"),
				daq.ToggleSwitch(id='run', value=False),

				html.Br(),
				dcc.Dropdown(id='color_by', style={'width': '100%'},
							 options=[{'label': i.title(), 'value': i} for i in mob_modes], value='walk'),

				html.Br(),
				html.Br(),
				dash_table.DataTable(
					id='attribute_table',
					columns=([{"name": 'Feature', "id": 'Feature'}, {"name": 'Mean', "id": 'Mean'}, {"name": 'Max', "id": 'Max'}]),
					data=[],
					style_as_list_view=True,
				),
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
	Output("attribute_table", "data"),
	Input("direct", "value"),
	Input("parcels", "value"),
	Input("buildings", "value"),
	Input("color_by", "value"),
	Input("run", "value"),
	Input("export", "value"),
	Input("click-info-json-output", "data")
)
def update_output(direct, parcels, buildings, color_by, run, export, memory):

	print(memory)

	tooltip = {
		"html": f"{color_by.title()} "+"{"f"{color_by}"+"}"
	}

	zero_margin = {'b': 0, 'l': 0, 'r': 0, 't': 0}
	color_map = {"drive": "D62728", "transit": "1F77B4", "walk": "2CA02C", "bike": "FF7F0E"}

	bus_routes = pdk.Layer(
		"GeoJsonLayer",
		filter_buffer(str_gdf, 'Transit'),
		id='bus_routes',
		opacity=0.62,
		stroked=True,
		get_fill_color=[int(i * 255) for i in colors.to_rgb("#1F77B4")],
		get_line_color=[255, 255, 255]
	)

	bike_lanes = pdk.Layer(
		"GeoJsonLayer",
		filter_buffer(str_gdf, 'BikeLane'),
		id='bike_lanes',
		opacity=0.62,
		stroked=True,
		get_fill_color=[int(i * 255) for i in colors.to_rgb("#FF7F0E")],
		get_line_color=[255, 255, 255, 0]
	)

	r = pdk.Deck(
		layers=[bus_routes, bike_lanes],
		initial_view_state=pdk.ViewState(latitude=0, longitude=0, zoom=15, max_zoom=16, pitch=0, bearing=0),
		mapbox_key=api_keys['mapbox'],
		map_style=pdk.map_styles.LIGHT,
	)

	if run:
		predicted = predict_mobility(gpd.read_file(f"{direct}/{parcels}"), gpd.read_file(f"{direct}/{buildings}"),
		                             str_gdf, mob_modes, file_path=f'Regression/{sb_name}_{parcels}.feather', suffix=parcels)

	else:
		predicted = gpd.read_feather(f'Regression/{sb_name}_{parcels}.feather')

	predicted = predicted.to_crs(26910)
	predicted['geometry'] = predicted.buffer(1)
	predicted = predicted.to_crs(4326)

	# Load importance data
	importance = load_importance()
	if export: importance.to_csv(f'tables/importance_{sb_name}_{parcels}.feather.csv')
	importance = importance.groupby(['key', 'feature']).sum()
	importance  = importance.loc[color_by, :].sort_values('importance', ascending=False)
	prd = predicted.loc[:, list(importance.index)[:6]]
	att_table = pd.concat([pd.DataFrame(prd.mean()), pd.DataFrame(prd.max())], axis=1).round(2).reset_index()
	att_table.columns = ['Feature', 'Mean', 'Max']
	neigh = Neighbourhood(parcels=predicted.copy())

	# Get centroid
	centroid = predicted.unary_union.centroid

	mode = color_by
	predicted['active'] = predicted['walk'] + predicted['bike']
	predicted = get_rgb_values(predicted, mode, modes_colors[mode])

	lyr = Layers(neighborhoods=neigh)
	lyr.layers = {}
	lyr.export_choropleth_pdk(mode, color_map=modes_colors[mode], n_breaks=20)

	# Create pie chart with mode shares
	shares_df = pd.DataFrame()
	shares_df['names'] = [mode.title() for mode in mob_modes]
	shares_df['shares'] = list(predicted.loc[:, mob_modes].mean())
	shares = px.pie(shares_df, names='names', values='shares', color='names', hole=0.6, opacity=0.8,
					color_discrete_map={k.title(): v for k, v in color_map.items()})
	shares.update_layout(margin=zero_margin, legend=dict(orientation="h"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
	if export: shares.write_image(f'images/{sb_name} - {parcels} - Mode Shares.png')

	# Build histogram
	disaggregated = pd.DataFrame()
	prd_copy = predicted.copy()
	for mode in mob_modes:
		prd_copy['mode'] = mode
		prd_copy['share'] = prd_copy[mode]
		disaggregated = pd.concat([disaggregated, prd_copy])

	color_discrete_map = {mode: f"#{color_map[mode]}" for mode in mob_modes}
	hist = px.histogram(disaggregated, x='share', color='mode', color_discrete_map=color_discrete_map)
	hist.update_layout(margin=zero_margin, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
	hist.update_traces(opacity=0.6)
	if export: hist.write_image(f'images/{sb_name} - {parcels} - Mode Shares - Histogram.png')

	# Export individual histograms
	for mode in mob_modes:
		hist2 = px.histogram(predicted, x=mode, color_discrete_sequence=[f"#{color_map[mode]}"])
		hist2.update_layout(margin=zero_margin, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
		if export: hist2.write_image(f'images/{sb_name} - {parcels} - Mode Shares - Histogram ({mode.title()}).png')

	if color_by is not None:
		r.layers = [bus_routes, bike_lanes] + [lyr.layers[key][0] for key in lyr.layers.keys() if len(lyr.layers[key]) > 0]
		print(f"{color_by} - {[l.id for l in r.layers]} - {modes_colors[color_by].name}")

	print([layer.id for layer in r.layers])

	r.initial_view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=15, max_zoom=16, pitch=0, bearing=0)
	r.update()
	print(r.selected_data)
	print(r.update())

	dgl = dash_deck.DeckGL(r.to_json(), id="deck", mapboxKey=r.mapbox_key, tooltip=tooltip, enableEvents=['click'],
	                       style={'width': '100%', 'float': 'left', 'display': 'inline-block'},)
	gc.collect()
	if export: att_table.to_csv(f'tables/{sb_name}_{parcels}_Important_{color_by.title()}.csv')
	return [dgl, shares, hist, att_table.to_dict('records')]


if __name__ == "__main__":
	app.run_server(debug=False, host='localhost', port=7050)
