import geopandas as gpd
import matplotlib.pyplot as plt
from Analyst import GeoBoundary
import pandas as pd
import numpy as np
from geopy.distance import distance
from scipy.spatial.distance import cdist
from Geospatial.Converter import polygon_grid
import matplotlib.font_manager as fm
from matplotlib import rc

fm.fontManager.ttflist += fm.createFontList(['/Volumes/Samsung_T5/Fonts/roboto/Roboto-Light.ttf'])
rc('font', family='Roboto', weight='light')
directory = '/Volumes/Samsung_T5/Databases'
experiments = ['e0', 'e1', 'e2', 'e3']
modes = ['walk', 'bike', 'drive', 'bus']

# Calculate differences from E0
proxy_files = {
    'E0': gpd.read_file(f'{directory}/Network/Hillside Quadra Sandbox_mob_e0_na.geojson_s0.geojson'),
    'E1': gpd.read_file(f'{directory}/Network/Hillside Quadra Sandbox_mob_e1_na.geojson_s0.geojson'),
    'E2': gpd.read_file(f'{directory}/Network/Hillside Quadra Sandbox_mob_e2_na.geojson_s0.geojson'),
    'E3': gpd.read_file(f'{directory}/Network/Hillside Quadra Sandbox_mob_e3_na.geojson_s0.geojson'),
}

# Load blocks layer
blocks_gdf = gpd.read_file(f'{directory}/Sandbox/Hillside Quadra/Mode Shifts - Urban Blocks.geojson')
blocks_gdf.crs = 26910

# Spatial join from parcels to grid
proxy = GeoBoundary('Hillside Quadra Sandbox', crs=26910, directory='/Volumes/Samsung_T5/')

print("Joining resident counts from parcels to blocks")
for exp in experiments:
    gdf = proxy_files[exp.title()]
    gdf.columns = [col.lower() for col in gdf.columns]
    gdf[f'population_{exp}'] = gdf['population, 2016']
    blocks_gdf['block_id'] = blocks_gdf.index

    # Spatial join to blocks
    joined_population = gpd.sjoin(
        blocks_gdf, gdf.loc[:, [f'population_{exp}', 'geometry']])\
        .groupby('block_id', as_index=False).sum()
    # joined_mode_shares = gpd.sjoin(
    #     blocks_gdf, gdf.loc[:, [f"walk_{exp}_rf_n", f"bike_{exp}_rf_n", f"drive_{exp}_rf_n", f"bus_{exp}_rf_n", 'geometry']])\
    #     .groupby('block_id', as_index=False).median()

    # Merge to initial blocks layer
    blocks_gdf = blocks_gdf.merge(
        joined_population.loc[:, [f'population_{exp}', 'block_id']], on='block_id')

print("Estimating number of people that use each mode")
blocks_gdf.columns = [col.lower() for col in blocks_gdf.columns]
for mode in modes:

    # Iterate over experiments to calculate the number of people that shifted to each mode
    for exp in experiments:

        # Method based on mode shifts
        blocks_gdf[f"pop_{mode}_{exp}"] = blocks_gdf[f'population_{exp}'] * (1+(blocks_gdf[f'd_{exp}_{mode}']/100))

        # Method based on predicted mode share
        blocks_gdf[f"pop_{mode}_{exp}"] = blocks_gdf[f'population_{exp}'] * blocks_gdf[f'{mode}_{exp}_rf_n']

# Estimate emissions based on number of drivers and riders
td = 2.5 #km

# Load potential destinations from bc assessment
print("> Estimating travel demand")
crd_gdf = gpd.read_file(
    f'{directory}/Capital Regional District, British Columbia.gpkg', layer='land_assessment_fabric')
d_gdf = crd_gdf[(crd_gdf['n_use'] == 'CM') | (crd_gdf['n_use'] == 'MX')]
d_gdf = d_gdf.reset_index()

# Convert to WGS84
d_gdf_4326 = d_gdf.to_crs(4326)
blocks_gdf_4326 = blocks_gdf.to_crs(4326)

for exp in experiments:
    # Get destinations within the sandbox
    parcel_gdf = gpd.read_file(proxy.gpkg, layer=f'land_parcels_{exp.lower()}')
    sb_dst = parcel_gdf[(parcel_gdf['Landuse'] == 'CM') | (parcel_gdf['Landuse'] == 'MX')].to_crs(4326)
    final_dst = pd.concat([d_gdf_4326, sb_dst])

    # blocks_gdf_4326['lat'] = np.radians(blocks_gdf_4326['geometry'].centroid)
    # blocks_gdf_4326['lon'] = np.radians(blocks_gdf_4326['geometry'].centroid)

    # Get distance from blocks to all destinations
    if f'{exp}_td' not in blocks_gdf.columns:
        for i, pt0 in enumerate(blocks_gdf_4326['geometry'].centroid):
            distances = []
            for pt1 in list(final_dst['geometry'].centroid):
                distances.append(distance(pt0.coords[0][::-1], pt1.coords[0][::-1]).km)
            blocks_gdf.at[i, f'{exp}_td'] = sum(distances)/len(distances)

    blocks_gdf[f'{exp}_car_em'] = (blocks_gdf[f"pop_drive_{exp}"] * blocks_gdf[f'{exp}_td'] * 0.16 * 2 * 365) / blocks_gdf[f'population_{exp}']
    blocks_gdf[f'{exp}_bus_em'] = (blocks_gdf[f"pop_bus_{exp}"] * blocks_gdf[f'{exp}_td'] * 0.07 * 2 * 365) / blocks_gdf[f'population_{exp}']
    blocks_gdf[f'{exp}_total_em'] = blocks_gdf[f'{exp}_car_em'] + blocks_gdf[f'{exp}_bus_em']

# Plot results
blocks_gdf = blocks_gdf.to_crs(26910)
fig_size = (15, 10)
fig1, ax = plt.subplots(nrows=len(['drive', 'bus', 'total']), ncols=len(experiments), figsize=fig_size) # Emissions per mode
for i, exp in enumerate(experiments):
    print(f"\nPlotting results for {exp}")
    for j, (mode, cmap) in enumerate(zip(['car', 'bus', 'total'], ['Reds', 'Blues', 'Greys'])):

        # Calculate mean and median
        mean = blocks_gdf[f'{exp}_{mode}_em'].mean()

        # Plot block maps with emissions per each mode per capita
        cols = [f'{e}_{mode}_em' for e in experiments]
        print(f"> Plotting {mode} blocks for {exp}")
        vmin = min(blocks_gdf.loc[:, cols].min())
        vmax = max(blocks_gdf.loc[:, cols].max())
        blocks_gdf.plot(f"{exp}_{mode}_em", ax=ax[j][i], legend=True, vmin=vmin, vmax=vmax, cmap=cmap)
        ax[j][i].set_title(f"{exp.upper()}, {mode.upper()} | MEAN: {round(mean/1000, 2)} t/person/yr")
        ax[j][i].set_axis_off()

# Export plots and maps to files
"blocks_gdf.to_file('Maps/UrbanBlocks.shp', driver='ESRI Shapefile')"
plt.tight_layout()
fig1.savefig(f'Maps/Mode Shifts - Emissions per Capita.png')
print(f"End")
