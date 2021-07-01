import os
import geopandas as gpd
from Morphology.ShapeTools import Analyst


directory = '/Users/nicholasmartino/Desktop/old_versions'
grid = gpd.read_file('/Volumes/SALA/Research/eLabs/50_projects/20_City_o_Vancouver/SSHRC Partnership Engage/Sandbox/shp/MainSt/Experiment/_hexagonal_grid.geojson')
experiments = {}
for mode in ['walk', 'bike', 'transit', 'drive']:
	experiments[mode] = []
	for file in sorted(os.listdir(directory)):
		if '.feather' in file:
			gdf = gpd.read_feather(f'{directory}/{file}')
			experiment = file.split('_')[3].split('.shp')[0]
			gdf['experiment'] = experiment
			experiments[mode].append(gdf.loc[:, [mode, 'geometry', 'experiment']])

for mode, gdfs in experiments.items():
	for i, gdf in enumerate(gdfs):
		print(f'E{i} ({mode}) - {gdf[mode].mean()}')
		if i > 0:
			gdf[f'{mode}_shift'] = (gdf[mode] - gdfs[0][mode])/gdfs[0][mode]
			grid_gdf = Analyst(grid, gdf).spatial_join()
			grid_gdf[f'{mode}_shift'] = grid_gdf[f'{mode}_shift_mean']
			grid_gdf.loc[:, [f'{mode}_shift', 'geometry']].to_crs(4326).to_file(f'{directory}/Shifts_E{i}_{mode}.geojson', driver='GeoJSON')
