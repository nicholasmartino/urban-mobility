import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point, LineString
from UrbanZoning.City.Network import Streets
from UrbanZoning.City.Fabric import Buildings, Parcels, Neighbourhood
from skbio import diversity
from Morphology.ShapeTools import Shape, Analyst

print("\n> Performing simple union for district-wide layers")


def overlay_radius(sample_gpkg, boundary_gpkg, sample_layer, boundary_layer='land_municipal_boundary', crs=26910, max_na_radius=4800):
    loc_bdr = gpd.read_file(boundary_gpkg, layer=boundary_layer)
    loc_bdr = loc_bdr.to_crs(boundary_gpkg.crs)
    loc_bdr_b = gpd.GeoDataFrame(geometry=loc_bdr.buffer(max_na_radius))

    gdf = gpd.read_file(sample_gpkg, layer=sample_layer)
    try:
        gdf.to_crs(crs)
    except:
        gdf.crs = crs

    ovr = gpd.overlay(gdf, loc_bdr_b)
    ovr.to_file(sample_gpkg, layer=sample_layer)
    return ovr

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

        # Potentially a function of Network class
        nodes = gpd.GeoDataFrame(columns=['geometry'])
        for i, mg in enumerate(streets.geometry):
            if mg.__class__.__name__ == 'LineString': mg=[mg]
            for ln in mg:
                ni = i * 2
                uid = lambda t: str("".join([str(o).replace('.', '') for o in list(t)]))
                nodes.at[ni, 'osmid'] = uid(ln.coords[0])
                nodes.at[ni, 'geometry'] = Point(ln.coords[0])
                nodes.at[ni + 1, 'osmid'] = uid(ln.coords[1])
                nodes.at[ni + 1, 'geometry'] = Point(ln.coords[1])
                streets.at[i, 'osmid'] = int(i)
                streets.at[i, 'from'] = uid(ln.coords[0])
                streets.at[i, 'to'] = uid(ln.coords[1])
                streets.at[i, 'geometry'] = ln

        # Assign a number for every unique osmid string
        replacements = {un: i for i, un in enumerate(nodes['osmid'].unique())}
        nodes['osmid'] = nodes['osmid'].replace(replacements).astype(int)
        streets.loc[:, ['to', 'from']] = streets.loc[:, ['to', 'from']].replace(replacements).astype(int)

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
        streets.to_file(local_gbd.gpkg, layer='network_walk')
        streets.to_file(local_gbd.gpkg, layer='network_drive')
        streets.to_file(local_gbd.gpkg, layer='network_all')

    return local_gbd

def proxy_indicators(local_gbd, experiment, parcels=True, cycling=True, transit=True, morphology=True):

    exp = list(experiment.keys())[0]
    yr = list(experiment.values())[0]

    if parcels:
        print("> Joining attributes from buildings to parcels")
        buildings = gpd.read_file(local_gbd.gpkg, layer=f'fabric_buildings_{exp}')
        parcels2 = gpd.read_file(local_gbd.gpkg, layer=f'land_parcels_{exp}')
        parcels2['OBJECTID'] = [i for i in range(len(parcels2))]

        # Rename land use standards
        parcels2 = parcels2.replace({'RS_SF_A': 'SFA', 'RS_MF_L': 'MFL', 'RS_MF_H': 'MFH', 'RS_SF_D': 'SFD'})

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
        try: parcels2 = parcels2.drop('OBJECTID', axis=1)
        except: pass
        parcels2.to_file(local_gbd.gpkg, layer=f"land_parcels_{exp}", driver='GPKG')

        print("> Adapting parcels to dissemination area")
        dss_are = parcels2
        try: dss_are["population, 2016"] = pcl_bdg.sum()[res_count_col].values
        except: dss_are["population, 2016"] = pcl_bdg.sum()[res_count_col].values
        dss_are["population density per square kilometre, 2016"] = pcl_bdg.sum()[res_count_col].values / parcels2['area']

        n_dwell = ['n_res_unit', 'res_units_bdg', 'n_res_unit_bdg', 'res_units']
        for col in n_dwell:
            if col in pcl_bdg_raw.columns:
                dss_are["n_dwellings"] = pcl_bdg.sum()[col].values

        print("> Adapting parcels to assessment fabric")
        ass_fab = parcels2

        # ass_fab.loc[:, 'area'] = parcels2.loc[:, 'geometry'].area
        # ass_fab.loc[parcels2['area'] < 400, 'n_size'] = 'less than 400'
        # ass_fab.loc[(parcels2['area'] > 400) & (parcels2['area'] < 800), 'n_size'] = '400 to 800'
        # ass_fab.loc[(parcels2['area'] > 800) & (parcels2['area'] < 1600), 'n_size'] = '800 to 1600'
        # ass_fab.loc[(parcels2['area'] > 1600) & (parcels2['area'] < 3200), 'n_size'] = '1600 to 3200'
        # ass_fab.loc[(parcels2['area'] > 3200) & (parcels2['area'] < 6400), 'n_size'] = '3200 to 6400'
        # ass_fab.loc[parcels2['area'] > 6400, 'n_size'] = 'more than 6400'

        land_use = ['LANDUSE', 'Landuse_pcl', 'LANDUSE_pcl']
        for col in land_use:
            if col in pcl_bdg_raw.columns:
                ass_fab["n_use"] = [u[0] for u in pcl_bdg[col].unique()]
        ass_fab.loc[ass_fab['n_use'] == 'MX', 'n_use'] = 'CM' ### !!!

        # floor_area = ['floor_area', 'floor_area_bdg']
        # for col in floor_area:
        #     if col in pcl_bdg_raw.columns:
        #         ass_fab["total_finished_area"] = (pcl_bdg.sum()[col] * pcl_bdg.mean()['maxstories']).values
        #
        # # Get floor area from FAR if specific field does not exist
        # if len(set(pcl_bdg_raw.columns).intersection(floor_area)) == 0:
        #     ass_fab["total_finished_area"] = pcl_bdg['area'] * pcl_bdg['FAR']

        # ftprt_area = ['ftprt_area', 'Shape_Area_bdg']
        # for col in ftprt_area:
        #     if col in pcl_bdg_raw.columns:
        #         ass_fab["gross_building_area"] = (pcl_bdg.sum()[col] * pcl_bdg.mean()['maxstories']).values

        n_bed = ['n_bedrms', 'n_bedrms_bdg', 'num_bedrms_bdg', 'num_bedrms']
        for col in n_bed:
            if col in pcl_bdg_raw.columns:
                ass_fab["number_of_bedrooms"] = pcl_bdg.sum()[col].values

        ass_fab.to_file(local_gbd.gpkg, layer='land_assessment_fabric', encoding='ISO-8859-1')
        parcels2.to_file(local_gbd.gpkg, layer='land_assessment_parcels', encoding='ISO-8859-1')
        dss_are.to_file(local_gbd.gpkg, layer='land_dissemination_area', encoding='ISO-8859-1')

    # print("> Calculating diversity indices")

    # # Diversity of bedrooms
    # df_ddb = pd.DataFrame()
    # for u in range(1,5):
    #     if u < 4: df_ddb[f"{u}_bedrms"] = [len(buildings.loc[buildings[n_bedrms_col] == u])]
    #     elif u >= 4: df_ddb[f"4_plus_bedrms"] = [len(buildings.loc[buildings[n_bedrms_col] >= 4])]
    # dss_are["dwelling_div_bedrooms_si"] = [diversity.alpha_diversity("simpson", df_ddb)[0] for i in range(len(dss_are))]
    # dss_are["dwelling_div_bedrooms_sh"] = [diversity.alpha_diversity("shannon", df_ddb)[0] for i in range(len(dss_are))]

    # # Diversity of rooms
    # df_ddr = pd.DataFrame()
    # buildings['n_rooms'] = buildings['n_bedrms'] + buildings['n_baths'] + 2
    # for u in range(max(buildings['n_rooms'])):
    #     if u <= 4: df_ddr[f"less_4_rooms"] = [len(buildings.loc[buildings['n_rooms'] <= u])]
    #     elif (u > 4) and (u < 8): df_ddr[f"{u}_rooms"] = [len(buildings.loc[buildings['n_rooms'] == u])]
    #     else: df_ddr[f"8_plus_rooms"] = [len(buildings.loc[buildings['n_rooms'] == u])]
    # dss_are["dwelling_div_rooms_si"] = [diversity.alpha_diversity("simpson", df_ddr)[0] for i in range(len(dss_are))]
    # dss_are["dwelling_div_rooms_sh"] = [diversity.alpha_diversity("shannon", df_ddr)[0] for i in range(len(dss_are))]

    init_streets = gpd.read_file(local_gbd.gpkg, layer='network_walk')
    streets = init_streets

    if cycling:
        cycling_cols = [f'cycle_{yr}', 'BikeLane']
        for col in cycling_cols:
            if col in streets.columns:
                streets[f"cycle_{yr}"] = streets[col]

        streets["length"] = streets.geometry.length
        cycling = streets[streets[f"cycle_{yr}"] == 1]
        cycling['length'] = cycling.geometry.length

        if len(streets) < len(init_streets):
            print("!!! Streets line count smaller than initial !!!")

    if transit:
        print("> Joining transit frequency data")
        stops = gpd.GeoDataFrame({
            'geometry': [Point(geom.coords[0]) for geom in streets.geometry]
        }, geometry='geometry', crs=streets.crs)
        stops = stops.drop_duplicates(subset=['geometry']).reset_index(drop=True)

        if 'bus_2020' not in streets.columns:
            streets['bus_2020'] = np.nan

        if 'freqt_2020' not in streets.columns:
            streets['freqt_2020'] = np.nan

        if 'rapid_2020' not in streets.columns:
            try: streets['rapid_2020'] = streets['Transit']
            except: streets['rapid_2020'] = np.nan

        for freq in ['bus', 'freqt', 'rapid']:
            if f'{freq}_{yr}' not in streets.columns:
                streets[f'{freq}_{yr}'] = np.nan

        bus2020 = streets[streets['bus_2020'] == 1].geometry.buffer(5).unary_union
        freqt2020 = streets[streets['freqt_2020'] == 1].geometry.buffer(5).unary_union
        freqt2040 = streets[streets[f'freqt_{yr}'] == 1].geometry.buffer(5).unary_union
        rapid2020 = streets[streets['rapid_2020'] == 1].geometry.buffer(5).unary_union
        rapid2040 = streets[streets[f'rapid_{yr}'] == 1].geometry.buffer(5).unary_union

        frequencies = {'freqt': 2.2, 'rapid': 2, 'bus': 1.5}

        for i in list(stops.index):

            if stops.iloc[i]['geometry'].within(bus2020):
                stops.at[i, 'frequency'] = frequencies['bus']
                stops.at[i, 'frequency_2040'] = frequencies['bus']

            if stops.iloc[i]['geometry'].within(freqt2020):
                stops.at[i, 'frequency'] = frequencies['freqt']
                stops.at[i, 'frequency_2040'] = frequencies['freqt']

            if stops.iloc[i]['geometry'].within(rapid2020):
                stops.at[i, 'frequency'] = frequencies['rapid']
                stops.at[i, 'frequency_2040'] = frequencies['rapid']

            if stops.iloc[i]['geometry'].within(rapid2040) & (yr >= 2040):
                stops.at[i, 'frequency'] = frequencies['rapid']
                stops.at[i, 'frequency_2040'] = frequencies['rapid']

            if stops.iloc[i]['geometry'].within(freqt2040) & (yr >= 2040):
                stops.at[i, 'frequency'] = frequencies['freqt']
                stops.at[i, 'frequency_2040'] = frequencies['freqt']

        stops = stops.fillna(0)
        stops = stops[(stops['frequency'] > 0) | (stops['frequency_2040'] > 0)]
        stops.to_file(local_gbd.gpkg, layer=f'network_stops', encoding='ISO-8859-1')

    # Get morphological indicators
    if morphology:
        for name, gdf in {'walk': streets, 'drive': streets, 'bike': cycling}.items():
            streets = Streets(gdf)
            # streets.gdf = streets.dimension()
            # streets.gdf = streets.direction()
            streets.gdf[f'{name}_length'] = streets.gdf['length'].astype(int)
            # streets.gdf[f'{name}_straight'] = streets.gdf['straight']
            streets.gdf.to_file(local_gbd.gpkg, layer=f'network_{name}', encoding='ISO-8859-1')

    return local_gbd

def estimate_demand(origin_gdf, destination_gdf):
    # Load potential destinations
    print("> Estimating travel demand")
    crd_gdf = destination_gdf.copy()
    gdf = origin_gdf.copy()

    u_uses = list(crd_gdf['n_use'].unique())
    if ('MX' not in u_uses) or ('CM' not in u_uses):
        d_gdf = crd_gdf[crd_gdf['n_use'].isin(['entertainment', 'retail', 'office'])]
    else:
        d_gdf = crd_gdf[(crd_gdf['n_use'] == 'MX') | (crd_gdf['n_use'] == 'CM')]
    d_gdf = d_gdf.reset_index()

    # Make blocks GeoDataFrame
    blocks = gpd.GeoDataFrame({'geometry': [geom for geom in gdf.buffer(1).unary_union]})

    # Draw a line from each parcel to all destinations
    for p in list(blocks.index):
        ls = []
        for r in tqdm(list(d_gdf.index)):
            ls.append(LineString([gdf.iloc[p]['geometry'].centroid.coords[0],
                                  d_gdf.iloc[r]['geometry'].centroid.coords[0]]).length)
        blocks.at[p, 'demand'] = (sum(ls)/len(ls))/1000
    return blocks

def calculate_emissions(parcel_gdf, block_gdf, suffix='', directory='/Volumes/Samsung_T5/Databases'):
    # Iterate over each parcel to define the probability of dwellers to chose each mode
    agents = pd.DataFrame()
    gdf = parcel_gdf.copy()
    for i in list(gdf.index):
        if gdf.loc[i, 'population, 2016'] > 0:
            for j in range(int(gdf.loc[i, 'population, 2016'])):
                k = len(agents)
                agents.at[k, 'parcel'] = i
                agents.at[k, 'p_walk'] = gdf.loc[i, f'walk{suffix}']
                agents.at[k, 'p_bike'] = gdf.loc[i, f'bike{suffix}']
                agents.at[k, 'p_transit'] = gdf.loc[i, f'transit{suffix}']
                agents.at[k, 'p_drive'] = 1 - sum(gdf.loc[i, [f'walk{suffix}', f'bike{suffix}', f'transit{suffix}']])

    # Randomly chose the mode of each agent based on the probabilities
    print("> Assigning mode to each inhabitant")
    for k in list(agents.index):
        agents.at[k, 'mode'] = np.random.choice(
            ['walk', 'bike', 'transit', 'drive'], 1, p=list(agents.loc[k, ['p_walk', 'p_bike', 'p_transit', 'p_drive']]))[0]

    # Get number of people by mode choice by parcel
    for p in list(gdf.index):
        gdf.at[p, f'walkers{suffix}'] = len(agents[(agents['mode'] == 'walk') & (agents['parcel'] == p)])
        gdf.at[p, f'bikers{suffix}'] = len(agents[(agents['mode'] == 'bike') & (agents['parcel'] == p)])
        gdf.at[p, f'riders{suffix}'] = len(agents[(agents['mode'] == 'transit') & (agents['parcel'] == p)])
        gdf.at[p, f'drivers{suffix}'] = len(agents[(agents['mode'] == 'drive') & (agents['parcel'] == p)])

    # Join trip demand from blocks to parcels
    gdf = gpd.overlay(gdf, block_gdf.loc[:, ['geometry', 'demand']])

    # Estimate emissions for riders and drivers
    print("> Calculating potential emissions")
    transit_em = 70
    drive_em = 120
    gdf[f'transit_em{suffix}'] = transit_em * gdf['demand'] * gdf[f'riders{suffix}']
    gdf[f'drive_em{suffix}'] = drive_em * gdf['demand'] * gdf[f'drivers{suffix}']
    gdf[f'total_em_kg_trip{suffix}'] = (gdf[f'transit_em{suffix}'] + gdf[f'drive_em{suffix}'])/1000
    gdf[f'total_em_kg_yr{suffix}'] = gdf[f'total_em_kg_trip{suffix}'] * 2 * 200
    gdf[f'total_em_kg_yr_person{suffix}'] = gdf[f'total_em_kg_yr{suffix}']/gdf['population, 2016']

    return gdf

def bldgs_to_prcls(buildings_gdf, parcels_gdf):
    neigh = Neighbourhood(parcels=Parcels(parcels_gdf), buildings=Buildings(buildings_gdf))

    # Assign OBJECTID and parcel area
    parcels_gdf['OBJECTID'] = parcels_gdf.index
    parcels_gdf['parcelarea'] = parcels_gdf.area
    parcels_gdf['Shape_Area'] = parcels_gdf.area
    parcels_gdf['Shape_Leng'] = [geom.length for geom in Shape(parcels_gdf).min_rot_rec()['largest_segment']]
    parcels_gdf['element'] = 'Parcel'

    # Get land use from building
    to_join = ['LANDUSE', 'floor_area']
    buildings_gdf['floor_area'] = neigh.get_gfa()['gfa']
    overlay = gpd.overlay(parcels_gdf.drop(to_join, axis=1), buildings_gdf.loc[:, to_join + ['geometry']])
    parcels_gdf.loc[parcels_gdf['OBJECTID'].isin(overlay['OBJECTID']), to_join] = overlay.loc[:, to_join]

    # Classify shell types from land use
    for lu in parcels_gdf['LANDUSE'].unique():
        parcels_gdf.loc[parcels_gdf['LANDUSE'] == 'CM', 'shell_type'] = 'Retail'

    # Calculate FAR
    non_res = ['CM', 'IND']
    parcels_gdf['FAR'] = neigh.get_fsr()['fsr']
    parcels_gdf.loc[parcels_gdf['LANDUSE'] == 'CM', 'RFAR'] = parcels_gdf.loc[parcels_gdf['LANDUSE'] == 'CM', 'FAR']
    parcels_gdf.loc[parcels_gdf['LANDUSE'].isin(non_res), 'FAR'] = 0
    parcels_gdf.loc[parcels_gdf['LANDUSE'].isin(non_res), 'DDenp'] = 0
    parcels_gdf.loc[parcels_gdf['LANDUSE'].isin(non_res), 'res_units'] = 0

    return parcels_gdf
