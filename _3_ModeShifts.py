import gc
import os
import pandas as pd
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
    def __init__(self, baselines, scenarios, modes, random_seeds=1, directory=os.getcwd(), suffix='', c_maps=None):
        """
        :param scenarios: list of scenarios
        :param modes: list of modes
        :param random_seeds: number of random seeds that was ran for each scenario
        :param baselines: dict {'random_seed': gdf} of GeoDataFrames with column "{mode}_{scenario}_rf_{rs}_n" representing the mode shares
        :param directory: child directory to load and save files
        :param suffix: suffix when reading scenario files
        """

        self.exp = scenarios
        self.modes = modes
        self.r_seeds = random_seeds
        self.dir = directory
        if not os.path.exists(f'{self.dir}/ModeShifts'): os.mkdir(f'{self.dir}/ModeShifts')
        self.out_dir = f'{self.dir}/ModeShifts'
        self.baselines = baselines
        self.grid_gdf = polygon_grid(self.baselines[0])
        self.suffix = suffix
        self.fig_size = (3 * len(self.get_files(0).keys()), 12)
        self.cmaps = c_maps

        self.min = -30
        self.max = 30
        return

    def get_files(self, rs):
        return {exp: gpd.read_feather(f'{self.dir}/Regression/test_{exp}_s{rs}{self.suffix}.feather') for exp in self.exp}

    def mean_rs(self):
        """
        Get the mean of all random seeds
        :return:
        """
        return gdf.loc[:, [col for col in gdf.columns if f'{mode}_{exp}_rf' in col]].mean(axis=1)

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

                # Join baseline data
                for mode in self.modes:
                    proxy_gdf[f"{mode}_e0_rf_{rs}_n"] =  self.baselines[rs][f"{mode}_e0_rf_{rs}_n"]

                base_cols = [i for mode in self.modes for i in [f"{mode}_{exp}_rf_{rs}_n", f"{mode}_e0_rf_{rs}_n"]]
                grid_gdf = gpd.sjoin(
                    grid_gdf.loc[:, [col for col in grid_gdf if 'index_' not in col]],
                    proxy_gdf.loc[:, list(set(base_cols).difference(set(grid_gdf.columns)))+["geometry"]],
                    how='left'
                ).drop_duplicates('geometry')

                # Calculate delta from E0
                for mode in self.modes:
                    gdf[f"{mode}_{exp}_rf_{rs}_n"] = grid_gdf[f"{mode}_{exp}_rf_{rs}_n"]

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
        main = 9
        widths = [main for i in self.exp] + [1]
        heights = [main for i in self.modes]

        # Re-aggregate data from grid to blocks
        print("\nJoining results from grid to parcels and blocks")
        for rs in range(self.r_seeds):
            ax5 = {}
            fig1, ax = plt.subplots(nrows=len(self.modes), ncols=len(self.exp), figsize=self.fig_size)
            fig2, ax2 = plt.subplots(nrows=len(self.modes), ncols=len(self.exp), figsize=self.fig_size)
            fig5 = plt.figure(constrained_layout=True, figsize=self.fig_size)
            gs = fig5.add_gridspec(len(self.modes), (len(self.exp) + 1), width_ratios=widths, height_ratios=heights)

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
                    # vmin = min(grid_gdf.loc[:, cols].min())
                    # vmax = max(grid_gdf.loc[:, cols].max())
                    grid_gdf.plot(f"d_{exp}_{mode}_s{rs}", ax=ax2[j][i], legend=True, vmin=self.min, vmax=self.max, cmap=cmap)
                    ax2[j][i].set_title(f"{exp}, {mode.upper()} | {round(mean, 1)}%")
                    ax2[j][i].set_axis_off()

                    # Plot average grid maps
                    ax5[j][i] = fig5.add_subplot(gs[j, i])
                    all_mean = grid_gdf[f'd_{exp}_{mode}'].mean()
                    print(f"> Plotting {mode} raster for {exp}")
                    cols = [f"d_{e}_{mode}_s{rs}" for e in self.exp]
                    # vmin = min(grid_gdf.loc[:, cols].min())
                    # vmax = max(grid_gdf.loc[:, cols].max())
                    grid_gdf.plot(f"d_{exp}_{mode}", ax=ax5[j][i], legend=False, vmin=self.min, vmax=self.max, cmap=cmap)
                    ax5[j][i].set_title(f"{exp}, {mode.upper()} | MEAN: {round(all_mean, 1)}%")
                    ax5[j][i].set_axis_off()

                    # Plot colormap legend
                    if i == len(self.exp)-1:
                        ax5[j][i+1] = fig5.add_subplot(gs[j, i+1])
                        divider = make_axes_locatable(ax5[j][i+1])
                        leg_ax = divider.append_axes(position="right", size="100%", pad="0%", add_to_figure=False)
                        array = np.arange(self.min, self.max)
                        show = leg_ax.imshow([array], cmap=cmap, aspect='auto')
                        cb = fig5.colorbar(show, cax=ax5[j][i+1])
                        cb.set_label('Change from baseline (%)')
                        # ax5[j][i+1].set_axis_off()

            # Export plots and maps to files
            print("Saving blocks")
            # block_gdf.to_file(f'{directory}/Sandbox/Hillside Quadra/Urban Blocks - Seed {rs}.geojson', driver='GeoJSON')
            plt.tight_layout()
            fig1.savefig(f'{self.out_dir}/{sandbox} - Mode Shifts - Histogram - Seed {rs}.png')
            fig2.savefig(f'{self.out_dir}/{sandbox} - Mode Shifts - Raster Map - Seed {rs}.png')
            fig5.savefig(f'{self.out_dir}/{sandbox} - Mode Shifts - Raster Map - Mean.png')
            gc.collect()

        return grid_gdf

    def plot_blocks(self, block_gdf):
        grid_gdf = self.grid_gdf.copy()

        print("Iteration of random seeds finished, averaging an exporting results")
        all_cols = []
        for rs in range(self.r_seeds):
            for exp in self.exp:
                cols = [f"d_{exp}_{mode}_s{rs}" for mode in self.modes] + [f"{mode}_{exp}_rf_{rs}_n" for mode in self.modes]
                all_cols = all_cols + cols

        block_gdf['i'] = block_gdf.index
        b_geom = block_gdf['geometry']
        block_gdf = gpd.GeoDataFrame(
            gpd.sjoin(block_gdf, grid_gdf.loc[:, ['geometry'] + all_cols]).groupby('i', as_index=False).median())
        block_gdf = block_gdf.drop('index_right', axis=1)
        block_gdf['geometry'] = b_geom

        # Setup plot results
        main = 9
        widths = [main for i in self.exp] + [1]
        heights = [main for i in self.modes]

        fig4, ax4 = plt.subplots(nrows=len(self.modes), ncols=len(self.exp), figsize=self.fig_size)
        fig5 = plt.figure(constrained_layout=True, figsize=self.fig_size)
        gs = fig5.add_gridspec(len(self.modes), (len(self.exp) + 1), width_ratios=widths, height_ratios=heights)

        for rs in range(self.r_seeds):
            ax5 = {}

            cols = []
            for i, (exp, file) in enumerate(self.get_files(rs).items()):
                for j, (mode, cmap) in enumerate(zip(self.modes, self.cmaps)):
                    ax5[j] = {}

                    cols.append(f"d_{exp}_{mode}")
                    cols.append(f"{mode}_{exp}_rf_n")

                    # Calculate mean and median
                    mean = grid_gdf[f'd_{exp}_{mode}_s{rs}'].mean()
                    median = grid_gdf[f'd_{exp}_{mode}_s{rs}'].median()

                    # Plot block maps
                    print(f"> Plotting {mode} blocks for {exp} on random seed {rs}")
                    block_gdf.plot(f"d_{exp}_{mode}_s{rs}", ax=ax4[j][i], legend=True, vmin=self.min, vmax=self.max, cmap=cmap)
                    ax4[j][i].set_title(f"{exp}, {mode.upper()} | MEAN: {round(mean, 1)}%")
                    ax4[j][i].set_axis_off()

                    # Plot average grid maps
                    ax5[j][i] = fig5.add_subplot(gs[j, i])
                    all_mean = grid_gdf[f'd_{exp}_{mode}'].mean()
                    print(f"> Plotting {mode} raster for {exp}")
                    cols = [f"d_{e}_{mode}_s{rs}" for e in self.exp]
                    grid_gdf.plot(f"d_{exp}_{mode}", ax=ax5[j][i], legend=False, vmin=self.min, vmax=self.max, cmap=cmap)
                    ax5[j][i].set_title(f"{exp}, {mode.upper()} | MEAN: {round(all_mean, 1)}%")
                    ax5[j][i].set_axis_off()

                    # Plot colormap legend
                    if i == len(self.exp)-1:
                        ax5[j][i+1] = fig5.add_subplot(gs[j, i+1])
                        divider = make_axes_locatable(ax5[j][i+1])
                        leg_ax = divider.append_axes(position="right", size="100%", pad="0%", add_to_figure=False)
                        array = np.arange(self.min, self.max)
                        show = leg_ax.imshow([array], cmap=cmap, aspect='auto')
                        cb = fig5.colorbar(show, cax=ax5[j][i+1])
                        cb.set_label('Change from baseline (%)')
                        # ax5[j][i+1].set_axis_off()

            fig4.savefig(f'{self.out_dir}/{sandbox} - Mode Shifts - Block Map - Seed {rs}.png')
            fig5.savefig(f'{self.out_dir}/{sandbox} - Mode Shifts - Block Map - Mean.png')

        block_gdf = gpd.GeoDataFrame(
            gpd.sjoin(block_gdf, self.grid_gdf.loc[:, ['geometry'] + all_cols]).groupby('i', as_index=False).mean())
        block_gdf = block_gdf.drop('index_right', axis=1)
        block_gdf['geometry'] = b_geom
        block_gdf.to_file(f'{self.out_dir}/Mode Shifts - Urban Blocks.geojson', driver='GeoJSON')


for sandbox in experiments.keys():
    ms = ModeShifts(
        directory=f'{directory}Sandbox/{sandbox}',
        baselines={rs: gpd.read_feather(f'{directory}Sandbox/{sandbox}/Regression/test_e0_s{rs}_{sandbox}.feather') for rs in range(r_seeds)},
        scenarios=list(experiments[sandbox][1].keys())[1:],
        modes=modes,
        suffix=f'_{sandbox}',
        random_seeds=r_seeds,
        c_maps=['PiYG', 'PuOr', 'coolwarm']
    )
    ms.grid_gdf = ms.calculate_delta()

    # Sum walk and bike modes to get active transport shift
    for mode in modes:
        if mode in ['walk', 'bike']:
            for exp in ms.exp:
                for rs in range(ms.r_seeds):
                    ms.grid_gdf[f"d_{exp}_active_s{rs}"] = ms.grid_gdf[f"d_{exp}_walk_s{rs}"] + ms.grid_gdf[f"d_{exp}_bike_s{rs}"]

                    # Calculate average of all random seeds
                    ms.grid_gdf[f"active_{exp}_rf_n"] = ms.grid_gdf.loc[:, [col for col in ms.grid_gdf.columns if f'{mode}_{exp}_rf' in col]].mean(axis=1)
                    ms.grid_gdf[f'd_{exp}_active'] = ms.grid_gdf.loc[:, [col for col in ms.grid_gdf.columns if f'd_{exp}_{mode}' in col]].mean(axis=1)

    # Plot blocks
    ms.plot_blocks(block_gdf=gpd.read_file(f'{directory}Sandbox/{sandbox}/{sandbox} Sandbox.gpkg', layer='land_parcels_e0'))
