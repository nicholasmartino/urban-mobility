import geopandas as gpd
import matplotlib.pyplot as plt


BUFF_DIR = '/Volumes/ELabs/50_projects/20_City_o_Vancouver/SSHRC Partnership Engage/Sandbox/shp/MainSt/Network_buffers'
BUS = gpd.read_file(f'{BUFF_DIR}/bus_stops.shp')
BIKE = gpd.read_file(f'{BUFF_DIR}/bike_lanes.shp')
CIVIC = gpd.read_file(f'{BUFF_DIR}/civic.shp')
OPEN = gpd.read_file(f'{BUFF_DIR}/open_spaces.shp')
COMM0 = gpd.read_file(f'{BUFF_DIR}/comm_mx_E0.shp')
COMM1 = gpd.read_file(f'{BUFF_DIR}/comm_mx_E1.shp')
COMM2 = gpd.read_file(f'{BUFF_DIR}/comm_mx_E2.shp')
COMM3 = gpd.read_file(f'{BUFF_DIR}/comm_mx_E3.shp')

GPK = '/Volumes/Samsung_T5/Databases/Sandbox/Main Street/Main Street Sandbox.gpkg'
PRC0 = gpd.read_file(GPK, layer='land_parcels_sandbox_prcls_E0.shp')
PRC1 = gpd.read_file(GPK, layer='land_parcels_sandbox_prcls_E1.shp')
PRC2 = gpd.read_file(GPK, layer='land_parcels_sandbox_prcls_E2.shp')
PRC3 = gpd.read_file(GPK, layer='land_parcels_sandbox_prcls_E3.shp')

cm = [(PRC1, COMM1), (PRC2, COMM2), (PRC3, COMM3)]
COMM0['geometry'] = COMM0.buffer(1)

for i, (parcel, buffer) in enumerate(cm):
	i = i + 1
	parcel['id'] = parcel.reset_index(drop=False).index
	buffer = buffer[buffer['radius'] == 400]

	# Subtract E0 buffer from experiment buffer and find parcels within this intersection
	buffer = gpd.overlay(buffer, COMM0[COMM0['radius'] == 400].loc[:, ['geometry']], how='difference')
	inters = gpd.overlay(parcel, buffer.loc[:, ['geometry']])
	parcel_f = parcel[parcel['id'].isin(list(inters['id']))]
	parcel.loc[parcel['id'].isin(list(inters['id'])), 'gain_acc'] = 1

	# Export maps
	parcel.to_file(f'{BUFF_DIR}/added_e{i}.shp')
	fig, ax = plt.subplots(figsize=(9, 9))
	parcel.boundary.plot(ax=ax, alpha=0.7, color='gray')
	parcel_f.plot(ax=ax, alpha=0.7, color='#D57371')
	plt.axis('off')
	plt.title(f'Gained access to amenities in E{i}')
	fig.savefig(f'{BUFF_DIR}/added_e{i}.png', dpi=300)
