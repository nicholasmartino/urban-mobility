import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from skbio import diversity

def proxy_indicators(local_gbd, district_gbd, experiment, max_na_radius=4800):

    exp = list(experiment.keys())[0]
    yr = list(experiment.values())[0]

    loc_bdr = gpd.read_file(local_gbd.gpkg, layer='land_municipal_boundary')
    loc_bdr = loc_bdr.to_crs(local_gbd.crs)
    loc_bdr_b = gpd.GeoDataFrame(geometry=loc_bdr.buffer(max_na_radius))

    print("\n> Performing simple union for district-wide layers")
    def rd_repr_ovr_exp(left_gpkg, right_gpkg, layer, crs):
        gdf = gpd.read_file(left_gpkg, layer=layer)
        try: gdf.to_crs(crs)
        except: gdf.crs = crs
        ovr = gpd.overlay(gdf, loc_bdr_b)
        ovr.to_file(right_gpkg, layer=layer)
    for lyr in ['network_nodes', 'network_axial', 'network_drive', 'network_stops']:
        rd_repr_ovr_exp(district_gbd.gpkg, local_gbd.gpkg, layer=lyr, crs=local_gbd.crs)

    print("> Joining attributes from buildings to parcels")
    buildings = gpd.read_file(local_gbd.gpkg, layer=f'fabric_buildings_{exp}')
    parcels2 = gpd.read_file(local_gbd.gpkg, layer=f'land_parcels_{exp}')
    parcels2['OBJECTID'] = [i for i in range(len(parcels2))]

    if 'OBJECTID' not in parcels2.columns:
        print("!!! OBJECTID column not found on parcels !!!")

    # Join data from buildings to parcels
    pcl_bdg_raw = gpd.sjoin(parcels2, buildings, how='left', lsuffix="pcl", rsuffix="bdg")
    col2try = ["OBJECTID_pcl", "OBJECTID"]
    pcl_bdg = None
    while True:
        for col in col2try:
            try:
                pcl_bdg = pcl_bdg_raw.groupby(col)
                break
            except: pass
        break

    if pcl_bdg is None:
        print("!!! Grouped by parcel not defined !!!")

    parcels2['area'] = parcels2['geometry'].area
    parcels2["area_sqkm"] = parcels2['area'] / 1000000
    try:
        res_count_col = 'res_count'
        parcels2["population, 2016"] = pcl_bdg.sum()[res_count_col].values
    except:
        res_count_col = 'res_count_bdg'
        parcels2["population, 2016"] = pcl_bdg.sum()[res_count_col].values
    print(f"{exp} experiment with {parcels2['population, 2016'].sum()} people")
    parcels2.to_file(local_gbd.gpkg, layer=f"land_parcels_{exp}", driver='GPKG')

    print("> Adapting parcels to dissemination area")
    dss_are = parcels2
    try: dss_are["population, 2016"] = pcl_bdg.sum()[res_count_col].values
    except: dss_are["population, 2016"] = pcl_bdg.sum()[res_count_col].values
    dss_are["population density per square kilometre, 2016"] = pcl_bdg.sum()[res_count_col].values / parcels2['Shape_Area']
    try: dss_are["n_dwellings"] = pcl_bdg.sum()['n_res_unit'].values
    except: dss_are["n_dwellings"] = pcl_bdg.sum()['res_units_bdg'].values

    print("> Adapting parcels to assessment fabric")
    ass_fab = parcels2
    try: ass_fab["n_use"] = [u[0] for u in pcl_bdg['Landuse_pcl'].unique()]
    except: ass_fab["n_use"] = [u[0] for u in pcl_bdg['LANDUSE_pcl'].unique()]
    ass_fab.loc[:, 'area'] = parcels2.loc[:, 'geometry'].area
    ass_fab.loc[parcels2['area'] < 400, 'n_size'] = 'less than 400'
    ass_fab.loc[(parcels2['area'] > 400) & (parcels2['area'] < 800), 'n_size'] = '400 to 800'
    ass_fab.loc[(parcels2['area'] > 800) & (parcels2['area'] < 1600), 'n_size'] = '800 to 1600'
    ass_fab.loc[(parcels2['area'] > 1600) & (parcels2['area'] < 3200), 'n_size'] = '1600 to 3200'
    ass_fab.loc[(parcels2['area'] > 3200) & (parcels2['area'] < 6400), 'n_size'] = '3200 to 6400'
    ass_fab.loc[parcels2['area'] > 6400, 'n_size'] = 'more than 6400'
    try: ass_fab["total_finished_area"] = (pcl_bdg.sum()['floor_area'] * pcl_bdg.mean()['maxstories']).values
    except: ass_fab["total_finished_area"] = (pcl_bdg.sum()['floor_area_bdg'] * pcl_bdg.mean()['maxstories']).values
    try: ass_fab["gross_building_area"] = (pcl_bdg.sum()['ftprt_area'] * pcl_bdg.mean()['maxstories']).values
    except: ass_fab["gross_building_area"] = (pcl_bdg.sum()['Shape_Area_bdg'] * pcl_bdg.mean()['maxstories']).values
    try:
        for n_bedrms_col in ['n_bedrms', 'n_bedrms_bdg', 'num_bedrms_bdg']:
            try:
                ass_fab["number_of_bedrooms"] = pcl_bdg.sum()[n_bedrms_col].values
                break
            except: pass
    except: pass

    print("> Calculating diversity indices")

    # Diversity of bedrooms
    df_ddb = pd.DataFrame()
    for u in range(1,5):
        if u < 4: df_ddb[f"{u}_bedrms"] = [len(buildings.loc[buildings[n_bedrms_col] == u])]
        elif u >= 4: df_ddb[f"4_plus_bedrms"] = [len(buildings.loc[buildings[n_bedrms_col] >= 4])]
    dss_are["dwelling_div_bedrooms_si"] = [diversity.alpha_diversity("simpson", df_ddb)[0] for i in range(len(dss_are))]
    dss_are["dwelling_div_bedrooms_sh"] = [diversity.alpha_diversity("shannon", df_ddb)[0] for i in range(len(dss_are))]

    # # Diversity of rooms
    # df_ddr = pd.DataFrame()
    # buildings['n_rooms'] = buildings['n_bedrms'] + buildings['n_baths'] + 2
    # for u in range(max(buildings['n_rooms'])):
    #     if u <= 4: df_ddr[f"less_4_rooms"] = [len(buildings.loc[buildings['n_rooms'] <= u])]
    #     elif (u > 4) and (u < 8): df_ddr[f"{u}_rooms"] = [len(buildings.loc[buildings['n_rooms'] == u])]
    #     else: df_ddr[f"8_plus_rooms"] = [len(buildings.loc[buildings['n_rooms'] == u])]
    # dss_are["dwelling_div_rooms_si"] = [diversity.alpha_diversity("simpson", df_ddr)[0] for i in range(len(dss_are))]
    # dss_are["dwelling_div_rooms_sh"] = [diversity.alpha_diversity("shannon", df_ddr)[0] for i in range(len(dss_are))]

    init_streets = gpd.read_file(local_gbd.gpkg, layer='network_links')
    streets = init_streets
    streets["length"] = streets.geometry.length
    cycling = streets[streets[f"cycle_{yr}"] == 1]
    cycling['cycle_length'] = cycling.geometry.length

    print("> Joining transit frequency data")
    stops = gpd.GeoDataFrame({
        'geometry': [Point(geom.coords[0]) for geom in streets.geometry]
    }, geometry='geometry')
    stops = stops.drop_duplicates(subset=['geometry']).reset_index(drop=True)
    bus2020 = streets[streets['bus_2020'] == 1].geometry.buffer(5).unary_union
    freqt2040 = streets[streets['freqt_2040'] == 1].geometry.buffer(5).unary_union
    rapid2040 = streets[streets['rapid_2040'] == 1].geometry.buffer(5).unary_union
    for i in list(stops.index):
        if stops.iloc[i]['geometry'].within(bus2020):
            stops.at[i, 'frequency'] = 32  # Trips per day from 30 to 30 minutes
        if stops.iloc[i]['geometry'].within(rapid2040):
            stops.at[i, 'frequency'] = 48 # Trips per day from 7 to 7 minutes
        if stops.iloc[i]['geometry'].within(freqt2040) & yr == 2040:
            stops.at[i, 'frequency'] = 192 # Trips per day from 15 to 15 minutes
    stops = stops.fillna(0)
    stops = stops[stops['frequency'] > 0]

    if len(streets) < len(init_streets):
        print("!!! Streets line count smaller than initial !!!")

    stops.to_file(local_gbd.gpkg, layer='network_stops', encoding='ISO-8859-1')
    streets.to_file(local_gbd.gpkg, layer='network_links', encoding='ISO-8859-1')
    cycling.to_file(local_gbd.gpkg, layer='network_cycle', encoding='ISO-8859-1')
    ass_fab.to_file(local_gbd.gpkg, layer='land_assessment_fabric', encoding='ISO-8859-1')
    parcels2.to_file(local_gbd.gpkg, layer='land_assessment_parcels', encoding='ISO-8859-1')
    dss_are.to_file(local_gbd.gpkg, layer='land_dissemination_area', encoding='ISO-8859-1')

    return local_gbd

def proxy_network(local_gbd, run=True):
    """
    Creates to and from fields on line segments in order to perform network analysis
    :param local_gbd: class GeoBoundary
    :param run: boolean toggle to run function
    """

    if run:
        print("> Updating street network connectivity")
        streets_initial = gpd.read_file(local_gbd.gpkg, layer='network_links')
        streets = streets_initial.reset_index(drop=True)

        nodes = gpd.GeoDataFrame(columns=['geometry'])
        for i, mg in enumerate(streets.geometry):
            if mg.__class__.__name__ == 'LineString': mg=[mg]
            for ln in mg:
                ni = i * 2
                uid = lambda t: int("".join([str(o).replace('.', '') for o in list(t)]))
                nodes.at[ni, 'osmid'] = uid(ln.coords[0])
                nodes.at[ni, 'geometry'] = Point(ln.coords[0])
                nodes.at[ni + 1, 'osmid'] = uid(ln.coords[1])
                nodes.at[ni + 1, 'geometry'] = Point(ln.coords[1])
                streets.at[i, 'osmid'] = int(i)
                streets.at[i, 'from'] = uid(ln.coords[0])
                streets.at[i, 'to'] = uid(ln.coords[1])
                streets.at[i, 'geometry'] = ln

        nodes = nodes.drop_duplicates(subset='osmid', ignore_index=True)
        old_ids = nodes.osmid.unique()
        new_ids = [i for i in range(len(old_ids))]
        nodes = nodes.replace(old_ids, new_ids)
        streets = streets.replace(old_ids, new_ids)

        nodes.crs = local_gbd.crs
        streets.crs = local_gbd.crs

        if len(streets_initial) < len(streets):
            print("!!! Streets line count smaller than initial !!!")

        nodes.to_file(local_gbd.gpkg, layer='network_intersections')
        streets.to_file(local_gbd.gpkg, layer='network_links')

    return local_gbd

def estimate_emissions(gdf, title='', directory='/Volumes/Samsung_T5/Databases'):
    # Iterate over each parcel to define the probability of dwellers to chose each mode
    agents = pd.DataFrame()
    for i in list(gdf.index):
        if gdf.loc[i, 'population, 2016'] > 0:
            for j in range(int(gdf.loc[i, 'population, 2016'])):
                k = len(agents)
                agents.at[k, 'parcel'] = i
                agents.at[k, 'p_walk'] = gdf.loc[i, f'walk_{title}_n']
                agents.at[k, 'p_bike'] = gdf.loc[i, f'bike_{title}_n']
                agents.at[k, 'p_bus'] = gdf.loc[i, f'bus_{title}_n']
                agents.at[k, 'p_drive'] = gdf.loc[i, f'drive_{title}_n']

    # Randomly chose the mode of each agent based on the probabilities
    print("> Assigning mode to each inhabitant")
    for k in list(agents.index):
        agents.at[k, 'mode'] = np.random.choice(
            ['walk', 'bike', 'bus', 'drive'], 1, p=list(agents.loc[k, ['p_walk', 'p_bike', 'p_bus', 'p_drive']]))[0]

    # Get number of people by mode choice by parcel
    for p in list(gdf.index):
        gdf.at[p, 'walkers'] = len(agents[(agents['mode'] == 'walk') & (agents['parcel'] == p)])
        gdf.at[p, 'bikers'] = len(agents[(agents['mode'] == 'bike') & (agents['parcel'] == p)])
        gdf.at[p, 'riders'] = len(agents[(agents['mode'] == 'bus') & (agents['parcel'] == p)])
        gdf.at[p, 'drivers'] = len(agents[(agents['mode'] == 'drive') & (agents['parcel'] == p)])

    # Load potential destinations
    print("> Estimating travel demand")
    crd_gdf = gpd.read_file(
        f'{directory}/Capital Regional District, British Columbia.gpkg', layer='land_assessment_fabric')
    d_gdf = crd_gdf[
        (crd_gdf['n_use'] == 'retail') | (crd_gdf['n_use'] == 'office') | (crd_gdf['n_use'] == 'hospitality') |
        (crd_gdf['n_use'] == 'civic') | (crd_gdf['n_use'] == 'entertainment')]
    d_gdf = d_gdf.reset_index()

    # # Draw a line from each parcel to all destinations
    # for p in list(gdf.index):
    #     gdf.iloc[p]['geometry'].centroid.coords[0]
    #     ls = []
    #     for r in list(d_gdf.index):
    #         ls.append(LineString([gdf.iloc[p]['geometry'].centroid.coords[0],
    #                               d_gdf.iloc[r]['geometry'].centroid.coords[0]]).length)
    #         print(f"> Appended destination {r}/{len(d_gdf)-1} for {p}/{len(gdf)-1} proxy parcels")
    #     gdf.at[p, 'demand'] = (sum(ls)/len(ls))/1000
    gdf['demand'] = 5

    # Estimate emissions for riders and drivers
    print("> Calculating potential emissions")
    bus_em = 70
    drive_em = 120
    gdf['bus_em'] = bus_em * gdf['demand'] * gdf['riders']
    gdf['drive_em'] = drive_em * gdf['demand'] * gdf['drivers']
    gdf['total_em_kg_trip'] = (gdf['bus_em'] + gdf['drive_em'])/1000
    gdf['total_em_kg_yr'] = gdf['total_em_kg_trip'] * 2 * 200
    gdf['total_em_kg_yr_person'] = gdf['total_em_kg_yr']/gdf['population, 2016']

    return gdf