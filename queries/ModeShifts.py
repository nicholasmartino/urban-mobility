import geopandas as gpd

directory = '/Volumes/SALA/Research/eLabs/50_projects/20_City_o_Vancouver/SSHRC Partnership Engage/Sandbox/shp/MainSt/Experiment/Mode Shares'
sample = 'ModeShares__hexagonal_grid.geojson'
# sample = 'ModeShares_sandbox_prcls_E0.shp'

wlk0 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E0.shp_walk.feather')
dri0 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E0.shp_drive.feather')
bik0 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E0.shp_bike.feather')
tst0 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E0.shp_transit.feather')

wlk1 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E1.shp_walk.feather')
dri1 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E1.shp_drive.feather')
bik1 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E1.shp_bike.feather')
tst1 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E1.shp_transit.feather')

wlk2 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E2.shp_walk.feather')
dri2 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E2.shp_drive.feather')
bik2 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E2.shp_bike.feather')
tst2 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E2.shp_transit.feather')

wlk3 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E3.shp_walk.feather')
dri3 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E3.shp_drive.feather')
bik3 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E3.shp_bike.feather')
tst3 = gpd.read_feather(f'{directory}/{sample}_sandbox_prcls_E3.shp_transit.feather')

experiments = {
	'walk': [wlk0, wlk1, wlk2, wlk3],
	'drive': [dri0, dri1, dri2, dri3],
	'bike': [bik0, bik1, bik2, bik3],
	'transit': [tst0, tst1, tst2, tst3]
}

for mode, gdfs in experiments.items():
	for i, gdf in enumerate(gdfs):
		print(f'E{i} ({mode}) - {gdf[mode].mean()}')
		if i > 0:
			gdf[f'{mode}_shift'] = (gdf[mode] - gdfs[0][mode])/gdfs[0][mode]
			gdf.loc[:, [f'{mode}_shift', 'geometry']].to_file(f'{directory}/Shifts_E{i}_{mode}.geojson', driver='GeoJSON')

None
