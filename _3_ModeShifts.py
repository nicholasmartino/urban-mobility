import gc
import os
import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from UrbanScraper.Converter import polygon_grid
from _0_Variables import *
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable


fm.fontManager.ttflist += fm.createFontList(['/Volumes/Samsung_T5/Fonts/roboto/Roboto-Light.ttf'])
rc('font', family='Roboto', weight='light')


class ModeShifts:
    def __init__(self, baseline, scenarios, modes, random_seeds=1, directory=os.getcwd(), suffix=''):
        """
        :param scenarios: list of scenarios
        :param modes: list of modes
        :param r_seeds: number of random seeds that was ran for each scenario
        :param baseline: GeoDataFrame with column "{mode}_{scenario}_rf_{rs}_n" representing the mode shares
        :param directory: child directory to load and save files
        :param suffix: suffix when reading scenario files
        """

        self.baseline = baseline
        self.exp = scenarios
        self.modes = modes
        self.r_seeds = random_seeds
        self.dir = directory
        if not os.path.exists(f'{self.dir}/ModeShifts'): os.mkdir(f'{self.dir}/ModeShifts')
        self.out_dir = f'{self.dir}/ModeShifts'
        self.grid_gdf = polygon_grid(baseline)
        self.suffix = suffix
        self.fig_size = (3 * len(self.get_files(0).keys()), 12)
        self.cmaps = ['Purples', 'Greens', 'Reds', 'Blues']
        return

    def get_files(self, rs):
        return {exp: gpd.read_feather(f'{self.dir}/Regression/test_{exp}_s{rs}{self.suffix}.feather') for exp in self.exp}

    def calculate_delta(self):
        gdf = self.grid_gdf.copy()
        grid_gdf = self.grid_gdf.copy()

        # Spatial join from parcels to grid
        print("Joining from parcels to grid")
        for rs in range(self.r_seeds):
            # Generate proxy files
            proxy_files = self.get_files(rs)

            for exp, file in proxy_files.items():
                proxy_gdf = proxy_files[exp]
                proxy_gdf.crs = 26910

                grid_gdf = gpd.sjoin(
                    grid_gdf.loc[:, [col for col in grid_gdf if 'index_' not in col]],
                    proxy_gdf.loc[:, [f"{mode}_{exp}_rf_{rs}_n" for mode in self.modes]+["geometry"]],
                    how='inner'
                ).drop_duplicates('geometry')

                # Calculate delta from E0
                for mode in self.modes:
                    try: gdf[f"{mode}_{exp}_rf_{rs}_n"] = grid_gdf[f"{mode}_{exp}_rf_{rs}_n"]
                    except: pass
                    shift = ((grid_gdf[f"{mode}_{exp}_rf_{rs}_n"] - grid_gdf[f"{mode}_e0_rf_{rs}_n"]) / grid_gdf[f"{mode}_e0_rf_{rs}_n"]) * 100
                    grid_gdf[f"d_{exp}_{mode}_s{rs}"] = shift
                    gdf[f"d_{exp}_{mode}_s{rs}"] = shift

                    # Calculate average of all random seeds
                    gdf[f"{mode}_{exp}_rf_n"] = gdf.loc[:, [col for col in gdf.columns if f'{mode}_{exp}_rf' in col]].mean(axis=1)
                    gdf[f'd_{exp}_{mode}'] = gdf.loc[:, [col for col in gdf.columns if f'd_{exp}_{mode}' in col]].mean(axis=1)

        return gdf

    def plot_grids(self):
        grid_gdf = self.grid_gdf.copy()

        # Setup plot results
        fig1, ax = plt.subplots(nrows=len(self.modes), ncols=len(self.exp), figsize=self.fig_size)
        fig2, ax2 = plt.subplots(nrows=len(self.modes), ncols=len(self.exp), figsize=self.fig_size)
        fig5 = plt.figure(constrained_layout=True, figsize=self.fig_size)
        main = 9
        widths = [main for i in self.exp] + [1]
        heights = [main for i in self.modes]
        ax5 = {}
        gs = fig5.add_gridspec(len(self.modes), (len(self.exp) + 1), width_ratios=widths, height_ratios=heights)

        # Re-aggregate data from grid to blocks
        print("\nJoining results from grid to parcels and blocks")
        for rs in range(self.r_seeds):
            for i, (exp, file) in enumerate(self.get_files(rs).items()):

                print(f"> Joining {exp}")
                proxy_gdf = self.get_files(rs)[exp]
                proxy_gdf['i'] = proxy_gdf.index

                for j, (mode, cmap) in enumerate(zip(self.modes, self.cmaps)):
                    ax5[j] = {}
                    print(f"\nPlotting results for {mode}")

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
                    cols = [f"d_{e}_{mode}_s{rs}" for e in self.exp]
                    vmin = min(grid_gdf.loc[:, cols].min())
                    vmax = max(grid_gdf.loc[:, cols].max())
                    grid_gdf.plot(f"d_{exp}_{mode}_s{rs}", ax=ax2[j][i], legend=True, vmin=vmin, vmax=vmax, cmap=cmap)
                    ax2[j][i].set_title(f"{exp}, {mode.upper()} | {round(mean, 1)}%")
                    ax2[j][i].set_axis_off()

                    # Plot average grid maps
                    ax5[j][i] = fig5.add_subplot(gs[j, i])
                    all_mean = grid_gdf[f'd_{exp}_{mode}'].mean()
                    print(f"> Plotting {mode} raster for {exp}")
                    cols = [f"d_{e}_{mode}_s{rs}" for e in self.exp]
                    vmin = min(grid_gdf.loc[:, cols].min())
                    vmax = max(grid_gdf.loc[:, cols].max())
                    grid_gdf.plot(f"d_{exp}_{mode}", ax=ax5[j][i], legend=False, vmin=vmin, vmax=vmax, cmap=cmap)
                    ax5[j][i].set_title(f"{exp}, {mode.upper()} | MEAN: {round(all_mean, 1)}%")
                    ax5[j][i].set_axis_off()

                    if i == len(self.exp)-1:
                        ax5[j][i+1] = fig5.add_subplot(gs[j, i+1])
                        divider = make_axes_locatable(ax5[j][i+1])
                        leg_ax = divider.append_axes(position="right", size="100%", pad="0%", add_to_figure=False)
                        array = np.arange(vmin, vmax)
                        show = leg_ax.imshow([array], cmap=cmap, aspect='auto')
                        cb = fig5.colorbar(show, cax=ax5[j][i+1])
                        cb.set_label('Change from baseline (%)')
                        ax5[j][i+1].set_axis_off()

            # Export plots and maps to files
            print("Saving blocks")
            # block_gdf.to_file(f'{directory}/Sandbox/Hillside Quadra/Urban Blocks - Seed {rs}.geojson', driver='GeoJSON')
            plt.tight_layout()
            fig1.savefig(f'{self.out_dir}/Mode Shifts - Histogram - Seed {rs}.png')
            fig2.savefig(f'{self.out_dir}/Mode Shifts - Raster Map - Seed {rs}.png')
            fig5.savefig(f'{self.out_dir}/Mode Shifts - Raster Map - Mean.png')
            gc.collect()

        return grid_gdf

    def plot_blocks(self, block_gdf):
        grid_gdf = self.grid_gdf.copy()

        print("Iteration of random seeds finished, averaging an exporting results")
        b_geom = block_gdf['geometry']

        for rs in range(self.r_seeds):
            for exp in ['e0'] + self.exp:
                cols = [f"d_{exp}_{mode}_s{rs}" for mode in self.modes] + [f"{mode}_{exp}_rf_{rs}_n" for mode in self.modes]

        block_gdf['i'] = block_gdf.index
        b_geom = block_gdf['geometry']
        block_gdf = gpd.GeoDataFrame(
            gpd.sjoin(block_gdf, grid_gdf.loc[:, ['geometry'] + cols]).groupby('i', as_index=False).median())
        block_gdf = block_gdf.drop('index_right', axis=1)
        block_gdf['geometry'] = b_geom

        fig4, ax4 = plt.subplots(nrows=len(self.modes), ncols=len(self.exp), figsize=self.fig_size)
        for rs in range(self.r_seeds):
            cols = []
            for exp in ['e0'] + self.exp:
                for j, (mode, cmap) in enumerate(zip(self.modes, self.cmaps)):
                    cols.append(f"d_{exp}_{mode}")
                    cols.append(f"{mode}_{exp}_rf_n")

                    mean = grid_gdf[f'd_{exp}_{mode}_s{rs}'].mean()

                    # Plot block maps
                    print(f"> Plotting {mode} blocks for {exp} on random seed {rs}")
                    vmin = min(block_gdf.loc[:, cols].min())
                    vmax = max(block_gdf.loc[:, cols].max())
                    block_gdf.plot(f"d_{exp}_{mode}_s{rs}", ax=ax4[j][i], legend=True, vmin=vmin, vmax=vmax, cmap=cmap)
                    ax4[j][i].set_title(f"{exp}, {mode.upper()} | MEAN: {round(mean, 1)}%")
                    ax4[j][i].set_axis_off()

            fig4.savefig(f'{self.out_dir}/Mode Shifts - Block Map - Seed {rs}.png')

        block_gdf = gpd.GeoDataFrame(
            gpd.sjoin(block_gdf, self.grid_gdf.loc[:, ['geometry'] + cols]).groupby('i', as_index=False).mean())
        block_gdf = block_gdf.drop('index_right', axis=1)
        block_gdf['geometry'] = b_geom
        block_gdf.to_file(f'{self.out_dir}/Mode Shifts - Urban Blocks.geojson', driver='GeoJSON')


direct = f'/Volumes/Samsung_T5/Databases/Sandbox/'
for sandbox in experiments.keys():
    ms = ModeShifts(
        directory=f'{direct}{sandbox}',
        baseline=gpd.read_file(f'{direct}{sandbox}/Regression/test_e0_s0_{sandbox}.geojson'),
        scenarios=list(experiments[sandbox][1].keys()),
        modes=['walk', 'bike', 'drive', 'bus'],
        suffix=f'_{sandbox}',
        random_seeds=r_seeds
    )
    ms.grid_gdf = ms.calculate_delta()
    ms.plot_grids()
