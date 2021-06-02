import gc
import os
import pandas as pd
import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from UrbanScraper.Converter import polygon_grid
from SB0_Variables import *
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline as po
from datetime import datetime
from UrbanZoning.City.Fabric import Neighbourhood, Parcels


fm.fontManager.ttflist += fm.createFontList(['/Volumes/Samsung_T5/Fonts/roboto/Roboto-Light.ttf'])
rc('font', family='Roboto', weight='light')


class ModeShifts:
    def __init__(self, baseline, modes, block_gdf=None, random_seeds=1, directory=os.getcwd(), suffix='', c_maps=None, plot=True, memory=False, shares_gdfs=None, city_name=None):
        """
        :param scenarios: list of scenarios
        :param modes: list of modes
        :param random_seeds: number of random seeds that was ran for each scenario
        :param baselines: dict {'random_seed': gdf} of GeoDataFrames with column "{mode}_{scenario}_rf_{rs}_n" representing the mode shares
        :param directory: child directory to load and save files
        :param suffix: suffix when reading scenario files
        """

        self.exp = shares_gdfs.keys()
        self.modes = modes
        self.r_seeds = random_seeds
        self.dir = directory
        if not os.path.exists(f'{self.dir}/ModeShifts'): os.mkdir(f'{self.dir}/ModeShifts')
        self.out_dir = f'{self.dir}/ModeShifts'
        self.baseline = baseline
        self.grid_gdf = polygon_grid(self.baseline, cell_size=30)
        self.suffix = suffix
        # self.fig_size = (3 * len(self.get_files(0).keys()), 12)
        self.cmaps = c_maps
        self.plot = plot
        self.min = -30
        self.max = 30
        self.memory = memory
        self.shares_gdfs = shares_gdfs
        self.city_name = city_name
        self.block = block_gdf
        return

    def get_files(self, rs):
        return {exp: gpd.read_feather(f'{self.dir}/Regression/test_{exp}_s{rs}{self.suffix}.feather') for exp in self.exp}

    def mean_rs(self):
        """
        Get the mean of all random seeds
        :return:
        """
        gdf = self.grid_gdf.copy()

        for exp in self.exp:
            for mode in self.modes:
                d_gdf = gdf.loc[:, [col for col in gdf.columns if f'd_{mode}_{exp}' in col]]
                rf_gdf = gdf.loc[:, [col for col in gdf.columns if (f'{mode}_{exp}_rf' in col) & ('_n' in col)]]
                gdf[f"{mode}_{exp}"] = rf_gdf.mean(axis=1)
                gdf[f'{mode}_{exp}_max'] = rf_gdf.max(axis=1)
                gdf[f'{mode}_{exp}_min'] = rf_gdf.min(axis=1)
                gdf[f'{mode}_{exp}_med'] = rf_gdf.median(axis=1)

                gdf[f'd_{mode}_{exp}'] = d_gdf.mean(axis=1)
                gdf[f'd_{mode}_{exp}_max'] = d_gdf.max(axis=1)
                gdf[f'd_{mode}_{exp}_min'] = d_gdf.min(axis=1)
                gdf[f'd_{mode}_{exp}_med'] = d_gdf.median(axis=1)

        return gdf

    def calculate_delta(self, da_baseline=False):
        gdf = self.grid_gdf.copy()
        grid_gdf = self.grid_gdf.copy()

        # Spatial join from parcels to grid
        print("Joining from parcels to grid")
        for rs in range(self.r_seeds):
            # Generate proxy files
            if self.shares_gdfs is None: proxy_files = self.get_files(rs)
            else: proxy_files = self.shares_gdfs

            for exp, file in proxy_files.items():
                if self.shares_gdfs is None: proxy_gdf = proxy_files[exp]
                else:
                    try: proxy_gdf = self.shares_gdfs[exp][rs]
                    except: proxy_gdf = self.shares_gdfs[exp]
                proxy_gdf.crs = 26910

                # Join geometry from block layer if it doesn't exist
                if 'geometry' not in proxy_gdf.columns:
                    proxy_gdf['geometry'] = self.block['geometry']

                # Join baseline data
                for mode in self.modes:
                    proxy_gdf[f"{mode}_e0_rf_{rs}_n"] =  self.baseline[f"{mode}_rf_{rs}_n"]
                    proxy_gdf[f"{mode}_{exp}_rf_{rs}_n"] = proxy_gdf[f"{mode}_rf_{rs}_n"]
                    proxy_gdf = proxy_gdf.drop(f"{mode}_rf_{rs}_n", axis=1)

                base_cols = [i for mode in self.modes for i in [f"{mode}_{exp}_rf_{rs}_n", f"{mode}_e0_rf_{rs}_n"]]
                grid_gdf = gpd.sjoin(
                    grid_gdf.loc[:, [col for col in grid_gdf if 'index_' not in col]],
                    proxy_gdf.loc[:, list(set(base_cols).difference(set(grid_gdf.columns)))+["geometry"]],
                    how='left'
                ).drop_duplicates('geometry')

                # Calculate delta from E0
                for mode in self.modes:
                    # grid_gdf[f"{mode}_{exp}_rf_{rs}_n"] = grid_gdf[f"{mode}_rf_{rs}_n"]

                    shift = ((grid_gdf[f"{mode}_{exp}_rf_{rs}_n"] - grid_gdf[f"{mode}_e0_rf_{rs}_n"]) / grid_gdf[f"{mode}_e0_rf_{rs}_n"])
                    grid_gdf[f"d_{mode}_{exp}_s{rs}"] = shift

            for mode in self.modes:
                grid_gdf[f'd_{mode}_e0'] = 0

        gdf = grid_gdf.copy()
        # Calculate average of random seeds on E0
        for mode in self.modes:
            gdf[f"{mode}_e0"] = grid_gdf.loc[:, [col for col in grid_gdf.columns if f'{mode}_e0' in col]].mean(axis=1)

        if da_baseline:
            print("Getting baseline from dissemination areas")
            # Get baseline mode share from the real place
            gdf['id'] = gdf.index
            da = gpd.read_file(f"{directory}{self.city_name}.gpkg", layer='land_dissemination_area')
            overlay = gpd.sjoin(
                gdf.loc[:, [col for col in gdf.columns if col not in ['index_left', 'index_right']]],
                da.loc[:, ['walk', 'bike', 'bus', 'drive', 'geometry']]).groupby('id').mean()
            overlay['geometry'] = gdf['geometry']
            overlay['walk_e0'] = overlay['walk']
            overlay['bike_e0'] = overlay['bike']
            overlay['active_e0'] = overlay['walk'] + overlay['bike']
            overlay['transit_e0'] = overlay['bus']
            overlay['drive_e0'] = overlay['drive']
            overlay = pd.concat([overlay, gdf.loc[list(set(gdf.index).difference(set(overlay.index))), :]])
            print(f"{self.city_name}:\n{overlay.loc[:, ['walk_e0', 'bike_e0', 'transit_e0', 'drive_e0']].mean()}")

            # Sum mode share and mode shift
            for mode in self.modes:
                for exp in self.exp:
                    for rs in range(self.r_seeds):
                        overlay[f"{mode}_{exp}_rf_{rs}"] = (overlay[f"{mode}_e0"] * overlay[f"d_{mode}_{exp}_s{rs}"]) + overlay[f"{mode}_e0"]

            # Normalize mode shares to 0-1
            for exp in self.exp:
                for rs in range(self.r_seeds):
                    total = overlay.loc[:, [f"{mode}_{exp}_rf_{rs}" for mode in self.modes]].sum(axis=1)
                    for mode in self.modes: overlay[f"{mode}_{exp}_rf_{rs}_n"] = overlay[f"{mode}_{exp}_rf_{rs}"] / total

            if 'index_left' in overlay.columns: overlay = overlay.drop('index_left', axis=1)
            if 'index_right' in overlay.columns: overlay = overlay.drop('index_right', axis=1)

            overlay.crs = gdf.crs
            overlay = overlay.fillna(method='ffill')

            return overlay

        else:
            return gdf

    def get_all_data(self, emissions=False):
        all_shifts = pd.DataFrame()
        for j, mode in enumerate(self.modes):
            if mode == 'transit': people = 'riders'
            elif mode == 'drive': people = 'drivers'
            else: people = None
            for i, exp in enumerate(['e0'] + list(self.exp)):
                mode_shifts = pd.DataFrame()
                mode_shifts['Block'] = self.block.index
                mode_shifts['Experiment'] = exp
                mode_shifts['Mode'] = mode.title()
                mode_shifts[f'Share'] = self.block[f'{mode}_{exp}']
                mode_shifts['∆'] = self.block[f'd_{mode}_{exp}']
                mode_shifts['Order'] = i + j
                if emissions:
                    if mode in ['drive', 'transit']:
                        mode_shifts['Emissions'] = self.block[f'{mode}_em_{exp}']
                        mode_shifts['Emissions/Cap.'] = self.block[f'{mode}_em_{exp}']/self.block[f'{people}_{exp}']
                    else:
                        mode_shifts['Emissions'] = 0
                        mode_shifts['Emissions/Cap.'] = 0
                all_shifts = pd.concat([all_shifts, mode_shifts])
                all_shifts = all_shifts.fillna(0)
        return all_shifts

    def plot_grid_map(self):
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
                    mean = grid_gdf[f'd_{mode}_{exp}_s{rs}'].mean()
                    median = grid_gdf[f'd_{mode}_{exp}_s{rs}'].median()

                    if self.plot:
                        # Plot histograms
                        print(f"> Plotting {mode} histograms for {exp} on random seed {rs}")
                        ax[j][i].hist(grid_gdf[f"d_{mode}_{exp}_s{rs}"])
                        ax[j][i].set_title(f"{exp.upper()}, {mode.upper()}")
                        ax[j][i].axvline(mean, color='b', linestyle='--')
                        ax[j][i].axvline(median, color='b', linestyle='-')

                        # Plot grid maps
                        print(f"> Plotting {mode} raster for {exp} on random seed {rs}")
                        cols = [f"d_{e}_{mode}_s{rs}" for e in self.exp]
                        # vmin = min(grid_gdf.loc[:, cols].min())
                        # vmax = max(grid_gdf.loc[:, cols].max())
                        grid_gdf.plot(f"d_{mode}_{exp}_s{rs}", ax=ax2[j][i], legend=True, vmin=self.min, vmax=self.max, cmap=cmap)
                        ax2[j][i].set_title(f"{exp}, {mode.upper()} | {round(mean, 1)}%")
                        ax2[j][i].set_axis_off()

                        # Plot average grid maps
                        ax5[j][i] = fig5.add_subplot(gs[j, i])
                        all_mean = grid_gdf[f'd_{mode}_{exp}'].mean()
                        print(f"> Plotting {mode} raster for {exp}")
                        cols = [f"d_{e}_{mode}_s{rs}" for e in self.exp]
                        # vmin = min(grid_gdf.loc[:, cols].min())
                        # vmax = max(grid_gdf.loc[:, cols].max())
                        grid_gdf.plot(f"d_{mode}_{exp}", ax=ax5[j][i], legend=False, vmin=self.min, vmax=self.max, cmap=cmap)
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

            if self.plot:
                plt.tight_layout()
                fig1.savefig(f'{self.out_dir}/{sandbox} - Mode Shifts - Histogram - Seed {rs}.png')
                fig2.savefig(f'{self.out_dir}/{sandbox} - Mode Shifts - Raster Map - Seed {rs}.png')
                fig5.savefig(f'{self.out_dir}/{sandbox} - Mode Shifts - Raster Map - Mean.png')
            gc.collect()

        return grid_gdf

    def join_blocks(self):
        grid_gdf = self.grid_gdf.copy()

        if self.block is None:
            block_gdf = Neighbourhood(parcels=Parcels(pd.concat(self.shares_gdfs.values()))).generate_blocks()
        else:
            block_gdf = self.block.copy()

        print("Joining results to block")
        block_gdf['i'] = block_gdf.index
        b_geom = block_gdf['geometry']
        group_by = gpd.sjoin(block_gdf, grid_gdf.loc[:, ['geometry'] + [col for col in grid_gdf.columns if col not in block_gdf.columns]]).groupby('i', as_index=False)
        block_gdf = gpd.GeoDataFrame(group_by.mean())
        block_gdf = block_gdf.drop('index_right', axis=1)
        block_gdf['geometry'] = b_geom

        return block_gdf

    def plot_block_map(self):
        grid_gdf = self.grid_gdf.copy()
        block_gdf = self.join_blocks()

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

                    cols.append(f"d_{mode}_{exp}")
                    cols.append(f"{mode}_{exp}_rf_n")

                    # Calculate mean and median
                    mean = grid_gdf[f'd_{mode}_{exp}_s{rs}'].mean()
                    median = grid_gdf[f'd_{mode}_{exp}_s{rs}'].median()

                    if self.plot:
                        # Plot block maps
                        print(f"> Plotting {mode} blocks for {exp} on random seed {rs}")
                        block_gdf.plot(f"d_{mode}_{exp}_s{rs}", ax=ax4[j][i], legend=True, vmin=self.min, vmax=self.max, cmap=cmap)
                        ax4[j][i].set_title(f"{exp}, {mode.upper()} | MEAN: {round(mean, 1)}%")
                        ax4[j][i].set_axis_off()

                        # Plot average grid maps
                        ax5[j][i] = fig5.add_subplot(gs[j, i])
                        all_mean = grid_gdf[f'd_{mode}_{exp}'].mean()
                        print(f"> Plotting {mode} raster for {exp}")
                        cols = [f"d_{e}_{mode}_s{rs}" for e in self.exp]
                        grid_gdf.plot(f"d_{mode}_{exp}", ax=ax5[j][i], legend=False, vmin=self.min, vmax=self.max, cmap=cmap)
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

            if self.plot:
                fig4.savefig(f'{self.out_dir}/{sandbox} - Mode Shifts - Block Map - Seed {rs}.png')
                fig5.savefig(f'{self.out_dir}/{sandbox} - Mode Shifts - Block Map - Mean.png')

        block_gdf.crs = self.grid_gdf.crs
        # block_gdf = gpd.GeoDataFrame(
        #     gpd.sjoin(block_gdf, self.grid_gdf.loc[:, ['geometry'] + all_cols]).groupby('i', as_index=False).mean())
        # block_gdf = block_gdf.drop('index_right', axis=1)
        # block_gdf['geometry'] = b_geom
        block_gdf.to_file(f'{self.out_dir}/Mode Shifts - Urban Blocks.geojson', driver='GeoJSON')
        return block_gdf

    def plot_blocks_box(self):
        fig = px.box(
            data_frame=self.get_all_data(),
            x="Mode",
            y=f'∆',
            facet_col='Experiment',
            points='all'
        )

        fig.show()
        po.plot(fig, filename=f'{self.out_dir}/BoxPlot {datetime.now()}.html')
        return fig

    def get_pop_count(self):

        print("Joining resident counts from parcels to blocks")
        for exp in experiments:
            gdf = proxy_files[exp.title()]
            gdf.columns = [col.lower() for col in gdf.columns]
            gdf[f'population_{exp}'] = gdf['population, 2016']
            blocks_gdf['id'] = blocks_gdf.index

            # Spatial join to blocks
            joined_population = gpd.sjoin(
                blocks_gdf, gdf.loc[:, [f'population_{exp}', 'geometry']]) \
                .groupby('id', as_index=False).sum()

            # Merge to initial blocks layer
            blocks_gdf = blocks_gdf.merge(
                joined_population.loc[:, [f'population_{exp}', 'id']], on='id')

        print("Estimating number of people that use each mode")
        blocks_gdf.columns = [col.lower() for col in blocks_gdf.columns]
        for mode in modes:

            # Iterate over experiments to calculate the number of people that shifted to each mode
            for exp in experiments:
                # Method based on mode shifts
                blocks_gdf[f"pop_{mode}_{exp}"] = blocks_gdf[f'population_{exp}'] * (
                            1 + (blocks_gdf[f'd_{exp}_{mode}'] / 100))

                # Method based on predicted mode share
                blocks_gdf[f"pop_{mode}_{exp}"] = blocks_gdf[f'population_{exp}'] * blocks_gdf[f'{mode}_{exp}_rf_n']
        return blocks_gdf


def calculate_mode_shifts(base_gdf, city_name, shares_gdfs=None, da_baseline=False):
    print("Calculating mode shifts")
    ms = ModeShifts(
        baseline=base_gdf,
        modes=modes,
        # block_gdf=gpd.read_file(f'{directory}Sandbox/{sandbox}/{sandbox} Sandbox.gpkg', layer='land_parcels_e0'),
        random_seeds=r_seeds,
        plot=False,
        c_maps=['PiYG', 'PuOr', 'coolwarm'],
        shares_gdfs=shares_gdfs,
        city_name=city_name,
    )
    ms.grid_gdf = ms.calculate_delta(da_baseline=da_baseline)
    ms.grid_gdf = ms.mean_rs()
    ms.block = ms.join_blocks()

    active = False
    if active:
        # Sum walk and bike modes to get active transport shift
        for m in modes:
            if m in ['walk', 'bike']:
                for e in ms.exp:
                    for rs in range(ms.r_seeds):
                        ms.grid_gdf[f"d_active_{e}_s{rs}"] = ms.grid_gdf[f"d_{e}_walk_s{rs}"] + ms.grid_gdf[
                            f"d_{e}_bike_s{rs}"]

                        # Calculate average of all random seeds
                        d = ms.grid_gdf.loc[:, [col for col in ms.grid_gdf.columns if f'd_{e}_{m}' in col]]
                        rf = ms.grid_gdf.loc[:, [col for col in ms.grid_gdf.columns if f'{m}_{e}_rf' in col]]

                        ms.grid_gdf[f"active_{e}"] = rf.mean(axis=1)
                        ms.grid_gdf[f'active_{e}_max'] = rf.max(axis=1)
                        ms.grid_gdf[f'active_{e}_min'] = rf.min(axis=1)
                        ms.grid_gdf[f'active_{e}_med'] = rf.median(axis=1)

                        ms.grid_gdf[f'd_active_{e}'] = d.mean(axis=1)
                        ms.grid_gdf[f'd_active_{e}_max'] = d.max(axis=1)
                        ms.grid_gdf[f'd_active_{e}_min'] = d.min(axis=1)
                        ms.grid_gdf[f'd_active_{e}_med'] = d.median(axis=1)
    return ms

if __name__ == '__main__':
    all_data = pd.DataFrame()
    for sb in experiments.keys():
        ms = calculate_mode_shifts(sandbox=sb)
        df = ms.get_all_data()
        df['Sandbox'] = sb
        all_data = pd.concat([all_data, df])
    all_data.to_csv(f'{directory}Sandbox/ModeShifts.csv', index=False)
