import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import gc
from Geospatial.Converter import polygon_grid
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

fm.fontManager.ttflist += fm.createFontList(['/Volumes/Samsung_T5/Fonts/roboto/Roboto-Light.ttf'])
rc('font', family='Roboto', weight='light')
name = 'Hillside Quadra'
directory = f'/Volumes/Samsung_T5/Databases/Sandbox/{name}'
exps = ['E1', 'E2', 'E3']
modes = ['walk', 'bike', 'drive', 'bus']

for infra in ['bike', 'bus']:
    block_gdf = gpd.read_file(f"{directory}/UrbanBlocks.shp")
    block_gdf.crs = 26910

    grid_gdf_raw = polygon_grid(gpd.read_file(f'{directory}/Hillside Quadra Sandbox_mob_e0_na.geojson_s0.geojson'))
    grid_gdf_all = grid_gdf_raw

    for rs in range(6):

        # Calculate differences from E0
        proxy_files = {
            'E0': gpd.read_file(f'{directory}/{name} Sandbox_mob_{infra}_e0_na.geojson_{infra}_s{rs}.geojson'),
            'E1': gpd.read_file(f'{directory}/{name} Sandbox_mob_{infra}_e1_na.geojson_{infra}_s{rs}.geojson'),
            'E2': gpd.read_file(f'{directory}/{name} Sandbox_mob_{infra}_e2_na.geojson_{infra}_s{rs}.geojson'),
            'E3': gpd.read_file(f'{directory}/{name} Sandbox_mob_{infra}_e3_na.geojson_{infra}_s{rs}.geojson'),
        }
        grid_gdf = grid_gdf_raw
        grid_gdf.crs = 26910

        # Spatial join from parcels to grid
        print("Joining from parcels to grid")
        for exp, file in proxy_files.items():
            proxy_gdf = proxy_files[exp]
            proxy_gdf.crs = 26910
            for i in ['index_left', 'index_right']:
                try: grid_gdf = grid_gdf.drop(i, axis=1)
                except: pass
            grid_gdf = gpd.sjoin(grid_gdf, proxy_gdf.loc[:,
                [f"walk_{exp}_rf_{rs}_n", f"bike_{exp}_rf_{rs}_n", f"drive_{exp}_rf_{rs}_n", f"bus_{exp}_rf_{rs}_n", "geometry"]], how='inner').drop_duplicates('geometry')

            # Calculate delta from E0
            for mode in modes:
                try: grid_gdf_all[f"{mode}_{exp}_rf_{rs}_n"] = grid_gdf[f"{mode}_{exp}_rf_{rs}_n"]
                except: pass
                shift = ((grid_gdf[f"{mode}_{exp}_rf_{rs}_n"] - grid_gdf[f"{mode}_E0_rf_{rs}_n"]) / grid_gdf[f"{mode}_E0_rf_{rs}_n"]) * 100
                grid_gdf[f"d_{exp}_{mode}_s{rs}"] = shift
                grid_gdf_all[f"d_{exp}_{mode}_s{rs}"] = shift

                # Calculate average of all random seeds
                grid_gdf_all[f"{mode}_{exp}_rf_n"] = grid_gdf_all.loc[:, [col for col in grid_gdf_all.columns if f'{mode}_{exp}_rf' in col]].mean(axis=1)
                grid_gdf_all[f'd_{exp}_{mode}'] = grid_gdf_all.loc[:, [col for col in grid_gdf_all.columns if f'd_{exp}_{mode}' in col]].mean(axis=1)

        # Re-aggregate data from grid to blocks
        print("\nJoining results from grid to parcels and blocks")
        for exp, file in proxy_files.items():

            print(f"> Joining {exp}")
            proxy_gdf = proxy_files[exp]
            proxy_gdf['i'] = proxy_gdf.index
            block_gdf['i'] = block_gdf.index
            cols = [f"d_{exp}_{mode}_s{rs}" for mode in modes]+[f"{mode}_{exp}_rf_{rs}_n" for mode in modes]

            b_geom = block_gdf['geometry']
            block_gdf = gpd.GeoDataFrame(gpd.sjoin(block_gdf, grid_gdf.loc[:, ['geometry']+cols]).groupby('i', as_index=False).median())
            block_gdf = block_gdf.drop('index_right', axis=1)
            block_gdf['geometry'] = b_geom

        # Plot results
        fig_size = (10, 12)
        fig1, ax = plt.subplots(nrows=len(modes), ncols=len(exps), figsize=fig_size)
        fig2, ax2 = plt.subplots(nrows=len(modes), ncols=len(exps), figsize=fig_size)
        fig3, ax3 = plt.subplots(nrows=len(modes), ncols=len(exps), figsize=fig_size)
        fig4, ax4 = plt.subplots(nrows=len(modes), ncols=len(exps), figsize=fig_size)
        fig5 = plt.figure(constrained_layout=True, figsize=fig_size)
        main = 9
        widths = [main, main, main, 1]
        heights = [main, main, main, main]
        ax5 = {}
        gs = fig5.add_gridspec(len(modes), (len(exps)+1), width_ratios=widths, height_ratios=heights)
        for j, (mode, cmap) in enumerate(zip(modes, ['Purples', 'Greens', 'Reds', 'Blues'])):
            ax5[j] = {}
            print(f"\nPlotting results for {mode}")
            for i, exp in enumerate(exps):
                bbox_to_anchor = (1, 0.5)

                # Calculate mean and median
                mean = grid_gdf[f'd_{exp}_{mode}_s{rs}'].mean()
                median = grid_gdf[f'd_{exp}_{mode}_s{rs}'].median()

                # Plot histograms
                print(f"> Plotting {mode} histograms for {exp} on random seed {rs}")
                ax[j][i].hist(grid_gdf[f"d_{exp}_{mode}_s{rs}"])
                ax[j][i].set_title(f"{exp.upper()}, {mode.upper()}")
                ax[j][i].axvline(mean, color='b', linestyle='--')
                ax[j][i].axvline(median, color='b', linestyle='-')

                # Plot grid maps
                print(f"> Plotting {mode} raster for {exp} on random seed {rs}")
                cols = [f"d_{e}_{mode}_s{rs}" for e in exps]
                vmin = min(grid_gdf.loc[:, cols].min())
                vmax = max(grid_gdf.loc[:, cols].max())
                grid_gdf.plot(f"d_{exp}_{mode}_s{rs}", ax=ax2[j][i], legend=True, vmin=vmin, vmax=vmax, cmap=cmap)
                ax2[j][i].set_title(f"{exp}, {mode.upper()} | {round(mean, 1)}%")
                ax2[j][i].set_axis_off()

                # Plot block maps
                print(f"> Plotting {mode} blocks for {exp} on random seed {rs}")
                vmin = min(block_gdf.loc[:, cols].min())
                vmax = max(block_gdf.loc[:, cols].max())
                block_gdf.plot(f"d_{exp}_{mode}_s{rs}", ax=ax4[j][i], legend=True, vmin=vmin, vmax=vmax, cmap=cmap)
                ax4[j][i].set_title(f"{exp}, {mode.upper()} | MEAN: {round(mean, 1)}%")
                ax4[j][i].set_axis_off()

                # Plot average grid maps
                ax5[j][i] = fig5.add_subplot(gs[j, i])
                all_mean = grid_gdf_all[f'd_{exp}_{mode}'].mean()
                print(f"> Plotting {mode} raster for {exp}")
                cols = [f"d_{e}_{mode}_s{rs}" for e in exps]
                vmin = min(grid_gdf.loc[:, cols].min())
                vmax = max(grid_gdf.loc[:, cols].max())
                grid_gdf_all.plot(f"d_{exp}_{mode}", ax=ax5[j][i], legend=False, vmin=vmin, vmax=vmax, cmap=cmap)
                ax5[j][i].set_title(f"{exp}, {mode.upper()} | MEAN: {round(all_mean, 1)}%")
                ax5[j][i].set_axis_off()

                if i == len(exps)-1:
                    ax5[j][i+1] = fig5.add_subplot(gs[j, i+1])
                    divider = make_axes_locatable(ax5[j][i+1])
                    leg_ax = divider.append_axes(position="right", size="100%", pad="0%", add_to_figure=False)
                    array = np.arange(vmin, vmax)
                    show = leg_ax.imshow([array], cmap=cmap, aspect='auto')
                    cb = fig5.colorbar(show, cax=ax5[j][i+1])
                    cb.set_label('Change from baseline (%)')

        # Export plots and maps to files
        print("Saving blocks")
        # block_gdf.to_file(f'{directory}/Sandbox/Hillside Quadra/Urban Blocks - Seed {rs}.geojson', driver='GeoJSON')
        plt.tight_layout()
        fig1.savefig(f'{directory}/Mode Shifts - {infra} - Histogram - Seed {rs}.png')
        fig2.savefig(f'{directory}/Mode Shifts - {infra} - Raster Map - Seed {rs}.png')
        fig4.savefig(f'{directory}/Mode Shifts - {infra} - Block Map - Seed {rs}.png')
        fig5.savefig(f'{directory}/Mode Shifts - {infra} - Raster Map - Mean.png')
        gc.collect()

    print("Iteration of random seeds finished, averaging an exporting results")
    b_geom = block_gdf['geometry']

    cols = []
    for exp in ['E0'] + exps:
        for mode in modes:
            cols.append(f"d_{exp}_{mode}")
            cols.append(f"{mode}_{exp}_rf_n")
    block_gdf = gpd.GeoDataFrame(
        gpd.sjoin(block_gdf, grid_gdf_all.loc[:, ['geometry'] + cols]).groupby('i', as_index=False).mean())
    block_gdf = block_gdf.drop('index_right', axis=1)
    block_gdf['geometry'] = b_geom

    block_gdf.to_file(f'{directory}/Mode Shifts - {infra.title()} - Urban Blocks.geojson', driver='GeoJSON')
    grid_gdf_all.to_file(f'{directory}/Mode Shifts - {infra.title()} - Grid.geojson', driver='GeoJSON')
    grid_gdf_all.to_file(f'{directory}/Mode Shifts - {infra.title()} - Grid.shp', driver='ESRI Shapefile')
