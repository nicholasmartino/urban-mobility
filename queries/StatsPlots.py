import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams["figure.autolayout"] = True


gdf = gpd.read_file('/Users/nicholasmartino/Desktop/Sunset_E0_Buildings.shp')
gdf = gdf[gdf['Total Ener'] <= 600]
gdf['TEUI (kwh/m2/year)'] = gdf['TEUI']
gdf['TEDI (kwh/m2/year)'] = gdf['TEDI']
gdf['Total Energy (mwh/year)'] = gdf['Total Ener']

columns = ['TEUI (kwh/m2/year)', 'TEDI (kwh/m2/year)', 'Total Energy (mwh/year)']
fig_height = 8

hist, h_ax = plt.subplots(1, len(columns), figsize=(fig_height * len(columns), fig_height))
maps, m_ax = plt.subplots(1, len(columns), figsize=(fig_height * len(columns), fig_height))
for i, col in enumerate(columns):
	sns.histplot(gdf, x=col, ax=h_ax[i])
	gdf.plot(col, ax=m_ax[i], legend=True)
	m_ax[i].set_axis_off()
	m_ax[i].set_title(col)

hist.savefig('histograms.png')
maps.savefig('maps.png')

columns = ['Total Energy (mwh/year)']
category = 'shell_type'
colors = ['#a6cee3', '#3dafd8', '#ff7f00', '#d5b43c', '#926234', '#5d5d5d', '#b93a3a', '#ffffad', '#1f78b4']

pies, p_ax = plt.subplots(1, len(columns), figsize=(fig_height * len(columns), fig_height))
for i, col in enumerate(columns):
	labels = list(gdf.groupby(category).groups.keys())
	if len(columns) > 1:
		axis = p_ax[i]
	else:
		axis = p_ax
	x = gdf.groupby(category)[col].sum() / gdf[col].sum()
	axis.pie(x=x, colors=colors)
	axis.set_title(col)

	pies.legend(labels=[f"{k} ({round(j, 1)} %)" for j, k in zip(x.values * 100, x.keys())], frameon=False, loc='upper left')
pies.savefig('pie_chart.png')
