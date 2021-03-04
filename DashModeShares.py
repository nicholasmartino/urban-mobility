# -*- coding: utf-8 -*-
import dash
import pandas as pd
import geopandas as gpd
import dash_daq as daq
import plotly.express as px
import json
import dash_table as dtb
import dash_deck
import pydeck as pdk
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from DashPredict_Back import analyze_sandbox, test_regression
from UrbanMobility.SB0_Variables import *
from matplotlib import colors, cm
from shapely.geometry import Polygon

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

BUILDINGS = "/Volumes/Samsung_T5/Databases/Sandbox/Main Street/buildings.geojson"
PARCELS = "/Volumes/ELabs/50_projects/20_City_o_Vancouver/SSHRC Partnership Engage/Sandbox/shp/MainSt/sandbox_demo_prcls_v3.shp"
STREETS = "/Volumes/ELabs/50_projects/20_City_o_Vancouver/SSHRC Partnership Engage/Sandbox/shp/MainSt/sandbox_demo_strts_segm.shp"

api_keys = {'mapbox': "pk.eyJ1IjoibmljaG9sYXNtYXJ0aW5vIiwiYSI6ImNrMjVhOGphOTAzZGUzbG8wNHJhdTZrMmYifQ.98uDMnGIvn1zrw4ZWUO35g"}
modes_colors = {'transit': cm.Blues, 'drive': cm.Reds, 'walk': cm.Greens, 'bike': cm.Oranges}
mob_modes = list(modes_colors.keys())

sb_name = 'Main Street'
pdk_layers = {}

def filter_buffer(gdf, layer):
	gdf = gdf[gdf[layer] == 1]
	gdf['geometry'] = gdf.buffer(5)
	return gdf.to_crs(4326)

def get_rgb_values(gdf, column, color_map):
	norm = colors.Normalize(vmin=min(gdf[column]), vmax=max(gdf[column]), clip=True)
	mapper = cm.ScalarMappable(norm=norm, cmap=color_map)
	gdf[f'{column}_colors'] = [[int(i * 255) for i in mapper.to_rgba(v)] for v in gdf[column]]
	return gdf

str_gdf = gpd.read_file(STREETS)
bus_routes = pdk.Layer(
	"GeoJsonLayer",
	filter_buffer(str_gdf, 'Transit')._to_geo(),
	id='bus_routes',
	opacity=0.62,
	stroked=True,
	get_fill_color=[int(i * 255) for i in colors.to_rgb("#1F77B4")],
	get_line_color=[255, 255, 255]
)

bike_lanes = pdk.Layer(
	"GeoJsonLayer",
	filter_buffer(str_gdf, 'BikeLane')._to_geo(),
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

tooltip = {
	"html": "Walk {walk} <br> Bike {bike} <br> Transit {transit} <br> Drive {drive}"
}

app.layout = \
	dbc.Col(
		style={'width': '100%', 'offset': 0, 'display': 'inline-block'},
		children=[
			dcc.Store(id='map_mode'),
			dcc.Store(id='memory'),
			html.Div(id="deck", style={'.bs-tooltip-left': {'top': '-12px', 'left': '-182px'}}),

			dbc.Col(style={'width': '300px', 'display': 'inline-block'}, className='pretty_container', children=[
				html.I("Define File Paths"),
				html.Br(),
				html.Br(),
				html.Label("Parcels"),
				dcc.Input(id="parcels", type="text", style={'width': '100%'},
						  value=PARCELS),
				html.Br(),
				html.Label("Buildings"),
				dcc.Input(id="buildings", type="text", style={'width': '100%'},
						  value=BUILDINGS),
				html.Br(),
				html.Label("Streets"),
				dcc.Input(id="streets", type="text", style={'width': '100%'},
						  value=STREETS),

				html.Br(),
				html.Br(),
				html.Label("Run"),
				daq.ToggleSwitch(id='run', value=False),

				html.Br(),
				# dcc.Dropdown(id='color_by', style={'width': '100%'},
				# 			 options=[{'label': i.title(), 'value': i} for i in mob_modes], value='walk'),

				html.Div(id="attribute_table"),
				dcc.Store(id="click-info-json-output"),
				html.Pre(id="click-event-json-output"),
			]),

			# dbc.Col(style={'width': '50%', 'display': 'inline-block'}, children=[
			#     dcc.Graph(id="map_view"),
			# ]),

			dbc.Col(style={'width': '20%', 'display': 'inline-block', 'float': 'right'}, className='pretty_container', children=[
				dbc.Row([
					dcc.Graph(id="hist", style={'width': '100%', 'height': '45vh'}),
				]),
				dbc.Row([
					dcc.Graph(id="shares", style={'width': '100%', 'height': '45vh'}),
				]),
			])
		])


def assign_callback(app, out, event):
	@app.callback(Output(f"attribute_table", "data"), [Input("deck", event)])
	def dump_json(data):

		data['object']['geometry'] = [Polygon(coord) for coord in data['object']['geometry']['coordinates']]
		data['object'] = {key: data['object'][key] for key in data['object'].keys() & {'walk', 'bike', 'transit', 'drive'}}
		df = pd.DataFrame(data['object'], index=[0])

		return dtb.DataTable(df.to_dict('records'))
assign_callback(app, "click-info", "clickInfo")


@app.callback(
	Output("deck", "children"),
	Output("shares", "figure"),
	Output("hist", "figure"),
	Input("parcels", "value"),
	Input("buildings", "value"),
	Input("streets", "value"),
	Input("run", "value"),
	# Input("color_by", "value"),
	Input("click-info-json-output", "data")
)
def update_output(parcels, buildings, streets, run, memory):

	print(memory)
	export=False

	zero_margin = {'b': 0, 'l': 0, 'r': 0, 't': 0}
	color_map = {"drive": "D62728", "transit": "1F77B4", "walk": "2CA02C", "bike": "FF7F0E"}

	if run:
		pcl_gdf = gpd.read_file(parcels)
		bdg_gdf = gpd.read_file(buildings)
		proxy_gdf = analyze_sandbox(bdg_gdf, pcl_gdf, str_gdf, sb_name='Main Street', suffix='_e0', ch_dir=directory)
		predicted = test_regression(proxy_gdf=proxy_gdf, label_cols=mob_modes, random_seeds=r_seeds).to_crs(4326) #.loc[:, mob_modes + ['geometry']].to_crs(4326)
		predicted = predicted.round(2)
		predicted.to_feather(f'Regression/{sb_name}.feather')
		predicted.to_file(f'Regression/{sb_name}.geojson', driver='GeoJSON')
	else:
		predicted = gpd.read_feather(f'Regression/{sb_name}.feather')

	predicted['geometry'] = predicted.buffer(0.00001)
	# predicted["coordinates"] = [[list(i) for i in list(geom.exterior.coords)] for geom in predicted['geometry']]

	# Get centroid
	centroid = predicted.unary_union.centroid

	# # Create map view using plotly choropleth mapbox
	# predicted['ID'] = predicted.index
	# map_view = px.choropleth_mapbox(
	# 	predicted, geojson=predicted._to_geo(), color=color_by, locations='ID', zoom=14, opacity=0.8,
	# 	range_color=[min(predicted[color_by]), max(predicted[color_by])], color_continuous_scale=modes_colors[color_by],
	# 	mapbox_style="carto-positron", center={"lat": centroid.y, "lon": centroid.x})
	# map_view.update_layout(margin=zero_margin)

	#
	color_by = 'walk'
	mode = color_by
	predicted['active'] = predicted['walk'] + predicted['bike']
	predicted = get_rgb_values(predicted, mode, modes_colors[mode])

	pdk_layers[mode] = pdk.Layer(
		"GeoJsonLayer",
		predicted.copy(),
		id=f"parcels_{mode}",
		opacity=0.62,
		stroked=False,
		filled=True,
		wireframe=True,
		get_fill_color=f"{mode}_colors",
		get_line_color=[255, 255, 255],
		auto_highlight=True,
		pickable=True)

	# Create pie chart with mode shares
	shares_df = pd.DataFrame()
	shares_df['names'] = [mode.title() for mode in mob_modes]
	shares_df['shares'] = list(predicted.loc[:, mob_modes].mean())
	shares = px.pie(shares_df, names='names', values='shares', color='names', hole=0.6, opacity=0.8,
					color_discrete_map={k.title(): v for k, v in color_map.items()})
	shares.update_layout(margin=zero_margin, legend=dict(orientation="h"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
	shares.write_image(f'images/{sb_name} - Mode Shares.png')

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
	if export: hist.write_image(f'images/{sb_name} - Mode Shares - Histogram.png')

	# Export individual histograms
	for mode in mob_modes:
		hist2 = px.histogram(predicted, x=mode, color_discrete_sequence=[f"#{color_map[mode]}"])
		hist2.update_layout(margin=zero_margin, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
		if export: hist2.write_image(f'images/{sb_name} - Mode Shares - Histogram ({mode.title()}).png')

	if color_by is not None:
		# Create RGB colors from color_by values

		# Prepare geometry for pydeck layer
		predicted['geometry'] = predicted.buffer(0.00001)
		# predicted["coordinates"] = [[list(i) for i in list(geom.exterior.coords)] for geom in predicted['geometry']]

		r.layers.append(pdk_layers[color_by])
		print(f"{color_by} - {[l.id for l in r.layers]} - {modes_colors[color_by].name}")

	print([layer.id for layer in r.layers])

	r.initial_view_state = 	pdk.ViewState(
		latitude=centroid.y, longitude=centroid.x, zoom=15, max_zoom=16, pitch=0, bearing=0
	)
	r.update()
	print(r.selected_data)
	print(r.update())
	return [
		dash_deck.DeckGL(
			r.to_json(), id="deck", mapboxKey=r.mapbox_key, tooltip=tooltip, enableEvents=['click']),
		shares, hist]


if __name__ == "__main__":
	app.run_server(debug=False, host='localhost')
