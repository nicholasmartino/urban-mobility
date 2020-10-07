import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Analyst import GeoBoundary
from geopy.distance import distance
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
pd.set_option('display.width', 700)

fm.fontManager.ttflist += fm.createFontList(['/Volumes/Samsung_T5/Fonts/roboto/Roboto-Light.ttf'])
rc('font', family='Roboto', weight='light')
name = 'Hillside Quadra'
directory = f'/Volumes/Samsung_T5/Databases/Sandbox/{name}'
experiments = ['e0', 'e1', 'e2', 'e3']
modes = ['walk', 'bike', 'drive', 'bus']
em_modes = ['car', 'bus', 'total']
exp_df = pd.DataFrame()
macc = pd.DataFrame()
tf_years = 20

for infra, values in {'bus': 'Frequent transit', 'bike': 'Cycling lanes'}.items():

    # Calculate differences from E0
    proxy_files = {
        'E0': gpd.read_file(f'{directory}/{name} Sandbox_mob_{infra}_e0_na.geojson_{infra}_s0.geojson'),
        'E1': gpd.read_file(f'{directory}/{name} Sandbox_mob_{infra}_e1_na.geojson_{infra}_s0.geojson'),
        'E2': gpd.read_file(f'{directory}/{name} Sandbox_mob_{infra}_e2_na.geojson_{infra}_s0.geojson'),
        'E3': gpd.read_file(f'{directory}/{name} Sandbox_mob_{infra}_e3_na.geojson_{infra}_s0.geojson'),
    }

    # Load blocks layer
    blocks_gdf = gpd.read_file(f'{directory}/Mode Shifts - {infra.title()} - Urban Blocks.geojson')
    blocks_gdf.crs = 26910

    # Spatial join from parcels to grid
    proxy = GeoBoundary('Hillside Quadra Sandbox', crs=26910, directory=directory)

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

    demand = False
    if demand:
        # Load potential destinations from bc assessment
        print("> Estimating travel demand")
        crd_gdf = gpd.read_file(
            f'/Volumes/Samsung_T5/Databases/Capital Regional District, British Columbia.gpkg',
            layer='land_assessment_fabric')
        d_gdf = crd_gdf[(crd_gdf['n_use'] == 'CM') | (crd_gdf['n_use'] == 'MX')]
        d_gdf = d_gdf.reset_index()

        # Convert to WGS84
        d_gdf_4326 = d_gdf.to_crs(4326)
        blocks_gdf_4326 = blocks_gdf.to_crs(4326)

        for i, exp in enumerate(experiments):
            # Get destinations within the sandbox
            parcel_gdf = gpd.read_file(proxy.gpkg, layer=f'land_parcels_{exp.lower()}')
            sb_dst = parcel_gdf[(parcel_gdf['Landuse'] == 'CM') | (parcel_gdf['Landuse'] == 'MX')].to_crs(4326)
            final_dst = pd.concat([d_gdf_4326, sb_dst])

            # Get distance from blocks to all destinations
            if f'{exp}_td' not in blocks_gdf.columns:
                for j, pt0 in enumerate(blocks_gdf_4326['geometry'].centroid):
                    distances = []
                    for pt1 in list(final_dst['geometry'].centroid):
                        distances.append(distance(pt0.coords[0][::-1], pt1.coords[0][::-1]).km)
                    blocks_gdf.at[j, f'{exp}_td'] = sum(distances)/len(distances)

            blocks_gdf[f'{exp}_car_em'] = (blocks_gdf[f"pop_drive_{exp}"] * blocks_gdf[f'{exp}_td'] * 0.16 * 3 * 365)
            blocks_gdf[f'{exp}_car_em_pc'] = blocks_gdf[f'{exp}_car_em']/blocks_gdf[f'population_{exp}']
            blocks_gdf[f'{exp}_bus_em'] = (blocks_gdf[f"pop_bus_{exp}"] * blocks_gdf[f'{exp}_td'] * 0.07 * 3 * 365)
            blocks_gdf[f'{exp}_bus_em_pc'] = blocks_gdf[f'{exp}_bus_em']/blocks_gdf[f'population_{exp}']
            blocks_gdf[f'{exp}_total_em'] = blocks_gdf[f'{exp}_car_em'] + blocks_gdf[f'{exp}_bus_em']
            blocks_gdf[f'{exp}_total_em_pc'] = blocks_gdf[f'{exp}_total_em']/blocks_gdf[f'population_{exp}']
            blocks_gdf = blocks_gdf.to_crs(26910)

        blocks_gdf.to_file(f'{directory}/UrbanBlocks - {infra.title()}.geojson', driver='GeoJSON')

    if not demand: blocks_gdf = gpd.read_file(f'{directory}/UrbanBlocks - {infra.title()}.geojson')

    # Estimate costs per mode on each experiment
    km_per_liter = 9
    price_per_liter = 1.2
    mtn_cost_per_km = 0.07 #https://driving.ca/auto-news/news/part-2-how-much-should-standard-car-maintenance-cost
    annual_insurance = 1400 #https://www.ratelab.ca/how-much-does-it-cost-to-own-and-operate-a-car-in-canada/
    cost_car_year_bc = 700 * 12 #City of Victoria, 2009. Go Victoria: Our Mobility Future
    transit_fare = 2.5
    bike_mtn_per_yr = 120 #https://bicycles.stackexchange.com/questions/15874/cost-of-maintenance-of-a-roadbike
    for i, exp in enumerate(experiments):

        # Calculate emissions/person saved from E0
        exp_df.at[i, f'{infra}_em_saved'] = blocks_gdf[f'e0_total_em'].sum() - blocks_gdf[f'{exp}_total_em'].sum()
        exp_df.at[i, f'{infra}_em_saved_per_inh'] = (blocks_gdf[f'e0_total_em'].sum() / blocks_gdf[f'population_e0'].sum()) - (
                    blocks_gdf[f'{exp}_total_em'].sum() / blocks_gdf[f'population_{exp}'].sum())

        for mode in modes:
            if mode == 'drive':
                # Cost based on individual estimates (Old version)
                blocks_gdf[f"cost_drive_{exp}"] = \
                    blocks_gdf[f"pop_drive_{exp}"] * annual_insurance * \
                    ((blocks_gdf[f'{exp}_td']/km_per_liter) * price_per_liter) * \
                        (blocks_gdf[f'{exp}_td'] * mtn_cost_per_km)
                # Cost based on aggregated data (City of Victoria, 2009)
                blocks_gdf[f"cost_drive_{exp}"] = cost_car_year_bc * blocks_gdf[f"pop_drive_{exp}"]
            elif mode == 'bus':
                blocks_gdf[f"cost_bus_{exp}"] = blocks_gdf[f"pop_bus_{exp}"] * transit_fare * 2 * 365
            elif mode == 'bike':
                blocks_gdf[f"cost_bike_{exp}"] = blocks_gdf[f"pop_bike_{exp}"] * bike_mtn_per_yr
            elif mode == 'walk':
                blocks_gdf[f"cost_walk_{exp}"] = 0
            exp_df.at[i, f"cost_{mode}"] = blocks_gdf[f"cost_{mode}_{exp}"].sum()

        # Aggregate costs per experiment
        exp_df["total_cost"] = exp_df[f"cost_bike"] + exp_df[f"cost_drive"] + exp_df[f"cost_bus"]
        exp_df.at[i, "pop"] = blocks_gdf[f'population_{exp}'].sum()
        exp_df["cost_per_cap"] = exp_df["total_cost"]/exp_df["pop"]

    # Infrastructure costs

    # https://www.victoria.ca/EN/main/residents/transportation/cycling/new-cycling-projects/funding.html
    bike_infra_cost_km = 100000

    # https://www.bctransit.com/documents/1529706193497
    vendor_pay = 69262676 + 11577349
    employee_net_pay = 43554814
    operations = 242546000
    maintenance = 55699000
    administration = 30419000
    total_fleet = 1+38+29+128+309+3+87+75+67+12+33+10+6+338+4 # https://bctransit.com/about/fleet
    amortization = 61047000
    debt = 9106000

    # https://bctransit.com/about/facts/victoria
    rev_fares = 37500000
    rev_pass = 4700000
    province = 4760000
    prop_tax = 30100000
    fuel_tax = 20400000
    adv = 700000
    total_trips = 27000000
    hours = 836000

    # Not sourced
    km_per_hour = 40
    km_routes_in_boundary = 1.6 # km

    km_travelled = hours * km_per_hour
    total_cost = operations + maintenance + administration
    total_revenue = rev_fares + rev_pass + province + prop_tax + fuel_tax + adv
    balance = total_revenue - total_cost
    balance_per_trip_per_km = (balance/total_trips/km_travelled) * -1

    local_gbd = GeoBoundary('Hillside Quadra Sandbox', crs=26910, directory='/Volumes/Samsung_T5/Databases/Sandbox/Hillside Quadra')

    links = gpd.read_file(local_gbd.gpkg, layer='network_links')
    stops = gpd.read_file(local_gbd.gpkg, layer='network_stops')
    for i, exp in enumerate(experiments):
        if exp == 'e0':
            yr = 2020
            exp_df.at[i, "cycling_cost"] = (links[links[f'cycle_2020'] == 1].length.sum() / 1000) * bike_infra_cost_km
        else:
            yr = 2040
            exp_df.at[i, "cycling_cost"] = (links[(links[f'cycle_2040'] == 1) & (links[f'cycle_2020'] == 0)].length.sum()/1000) * bike_infra_cost_km
        exp_df.at[i, "transit_cost"] = stops[f'frequency_{yr}'].sum() * balance_per_trip_per_km * km_routes_in_boundary * 365 * tf_years

    if infra == 'bus': infra_cost = exp_df["transit_cost"]
    elif infra == 'bike': infra_cost = exp_df["cycling_cost"]
    else: infra_cost = 0

    exp_df[f"{infra}_infra_cost"] = infra_cost
    exp_df[f"{infra}_infra_cost_per_cap"] = exp_df[f"{infra}_infra_cost"] / exp_df["pop"]

    # Plot results
    fig_size = (15, 10)
    fig1 = plt.figure(constrained_layout=False, figsize=fig_size)
    ax1 = {}
    main = 12
    widths = [main, main, main, main, 1]
    heights = [main, main, main]
    gs = fig1.add_gridspec(nrows=len(em_modes), ncols=len(experiments)+1, width_ratios=widths, height_ratios=heights)

    for i, (mode, cmap) in enumerate(zip(em_modes, ['Reds', 'Blues', 'viridis_r'])):
        ax1[i] = {}
        for j, exp in enumerate(experiments):
            print(f"\nPlotting results on {exp}(i: {i}) for {mode}(j: {j})")

            # Create non-NaN GeoDataFrame
            blocks_valid = blocks_gdf.dropna()

            # Calculate mean and median
            mean = blocks_valid[f'{exp}_{mode}_em_pc'].mean()

            # Get minimum and maximum values on each experiment
            cols = [f'{e}_{mode}_em_pc' for e in experiments]
            vmin = min(blocks_valid.loc[:, cols].min())
            vmax = max(blocks_valid.loc[:, cols].max())
            print(f"{cols} min: {vmin}, max: {vmax}, mean: {mean}")

            # Plot block maps with emissions per each mode per capita
            ax1[i][j] = fig1.add_subplot(gs[i, j])
            blocks_valid.plot(f"{exp}_{mode}_em_pc", ax=ax1[i][j], legend=False, vmin=vmin, vmax=vmax, cmap=cmap)
            ax1[i][j].set_title(f"{exp.upper()}, {mode.upper()} | MEAN: {round(mean/1000, 2)} tCO2/p/y")
            ax1[i][j].set_axis_off()

            if j == len(experiments) - 1:
                ax1[i][j + 1] = fig1.add_subplot(gs[i, j + 1])
                divider = make_axes_locatable(ax1[i][j + 1])
                leg_ax = divider.append_axes(position="bottom", size="100%", pad="0%", add_to_figure=False)
                array = np.arange(vmin, vmax)
                show = leg_ax.imshow([array], cmap=cmap, aspect='auto')
                cb = fig1.colorbar(show, cax=ax1[i][j + 1], orientation="vertical")
                cb.set_label('Emissions (tCO2/person/year)')

    # Export plots and maps to files
    plt.tight_layout()
    fig1.savefig(f'{directory}/Mode Shifts - Emissions per Capita - {infra}.png')

    # Export data on MACC DataFrame
    for i, exp in enumerate(experiments):
        if exp != 'e0':
            j = len(macc)
            macc.at[j, 'Abatement Measure'] = f'Densification {exp.title()} + {values}'
            macc.at[j, 'GHGs Abated (tCO2/per capita)'] = (exp_df.loc[i, f"{infra}_em_saved_per_inh"]/1000) * tf_years
            macc.at[j, 'GHGs Abated (tCO2)'] = (macc.at[j, 'GHGs Abated (tCO2/per capita)']) * exp_df.at[i, "pop"]
            macc.at[j, 'Net Cost of Abatement Measure ($)'] = exp_df.loc[i, f"{infra}_infra_cost"]

    macc['Marginal Cost ($/tCO2)'] = macc['Net Cost of Abatement Measure ($)']/macc['GHGs Abated (tCO2)']

print(macc)
print("Exporting MACC to excel")
macc.to_excel(f'{directory}/Mobility MACC.xlsx')
print(f"End")
