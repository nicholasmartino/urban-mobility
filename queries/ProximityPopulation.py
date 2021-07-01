import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from Proximity import Proximity
from UrbanZoning.City.Network import Streets


streets_gdf = gpd.read_file(
	f"/Volumes/Macintosh HD/Users/nicholasmartino/Google Drive/elementslab/main_st_streets_e0.geojson")
directory = '/Volumes/SALA/Research/eLabs/50_projects/20_City_o_Vancouver/SSHRC Partnership Engage/Sandbox/shp/MainSt/Experiment/Mode Shares'

output = pd.DataFrame()
for field, mode in {'BikeLane': 'bike', 'Transit': 'transit'}.items():
	### Filter streets GeoDataFrame
	assert field in streets_gdf.columns, KeyError(f"{field} column not found in streets_gdf")

	base_gdf = Streets(streets_gdf[streets_gdf[field] == 1]).segmentize()
	gdf_e0 = f'ModeShares_sandbox_prcls_E0.shp_sandbox_prcls_E0.shp_{mode}.feather'
	gdf_e1 = f'ModeShares_sandbox_prcls_E0.shp_sandbox_prcls_E1.shp_{mode}.feather'
	gdf_e2 = f'ModeShares_sandbox_prcls_E0.shp_sandbox_prcls_E2.shp_{mode}.feather'
	gdf_e3 = f'ModeShares_sandbox_prcls_E0.shp_sandbox_prcls_E3.shp_{mode}.feather'

	for i, file_name in enumerate([gdf_e0, gdf_e1, gdf_e2, gdf_e3]):
		### Get number of users
		assert file_name in os.listdir(directory), FileNotFoundError(f"{file_name} not found in {directory}")

		gdf = Proximity(base_gdf, gpd.read_feather(f'{directory}/{file_name}').loc[:, ['res_units', mode, 'geometry']].to_crs(26910)).get_proximity()
		base_gdf[f'{mode}_users_400m_e{i}'] = gdf['res_units_sum'] * gdf[f'{mode}_mean'] * 1.12

	output = pd.concat([output, base_gdf])

output.to_file('Transport_Users.geojson', driver='GeoJSON')

### Plot maps
for experiment in ['e0', 'e1', 'e2', 'e3']:
	fig, ax = plt.subplots(figsize=(15, 15))
	bike_output = output[~output[f'bike_users_400m_{experiment}'].isna()]
	transit_output = output[~output[f'transit_users_400m_{experiment}'].isna()]

	bike_output.plot(column=f'bike_users_400m_{experiment}', ax=ax, cmap='Oranges', linewidth=bike_output[f'bike_users_400m_{experiment}']/50)
	transit_output.plot(column=f'transit_users_400m_{experiment}', ax=ax, cmap='Blues', linewidth=transit_output[f'transit_users_400m_{experiment}']/50)
	plt.axis('off')
	fig.savefig(f'n_users_{experiment}.png')
