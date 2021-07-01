import dash_core_components as dcc
import dash_html_components as html
import os
from dash.dependencies import Input, Output
from SB0_Variables import *
from functions.dash import get_options


def settings_layout():
	return [html.Label('DIRECTORY'),
	dcc.Input(id='directory', value='/Users/nicholasmartino/Desktop/old_versions', style={'width': '100%'}),

	html.Br(),
	html.Br(),

	html.Label('BASELINE'),

	html.Br(),
	html.Label('Parcels'),
	html.Div(id='baseline_pcl'),

	html.Label('Buildings'),
	html.Div(id='baseline_bld'),

	html.Br(),
	html.Label('SCENARIOS'),
	html.Br(),

	html.Label('Parcels'),
	html.Div(id='scenarios_pcl'),

	html.Label('Buildings'),
	html.Div(id='scenarios_bld'),
	html.Br(),
	html.Br()]


def settings_callback(app):
	@app.callback(Output('scenarios_pcl', 'children'),
				  Output('baseline_pcl', 'children'),
				  Output('scenarios_bld', 'children'),
				  Output('baseline_bld', 'children'),
				  [Input('directory', 'value')])
	def update_menus(direct):
		if os.path.exists(direct):
			scenarios = dcc.Dropdown(id='scenarios_pcl', options=get_options(sorted(os.listdir(direct))), multi=True)
			baseline = dcc.Dropdown(id='baseline_pcl', options=get_options(sorted(os.listdir(direct)))),
			scenarios_b = dcc.Dropdown(id='scenarios_bld', options=get_options(sorted(os.listdir(direct))), multi=True)
			baseline_b = dcc.Dropdown(id='baseline_bld', options=get_options(sorted(os.listdir(direct))))
			return scenarios, baseline, scenarios_b, baseline_b
		else:
			print("Returning Markdowns...")
			scenarios = dcc.Markdown(id='scenarios_pcl')
			baseline = dcc.Markdown(id='baseline_pcl')
			scenarios_b = dcc.Markdown(id='scenarios_bld')
			baseline_b = dcc.Markdown(id='baseline_bld')
			return scenarios, baseline, scenarios_b, baseline_b
