import gc
import os
import pickle
import plotly.express as px
import time
import geopandas as gpd
import matplotlib.font_manager as fm
import pandas as pd
from Analyst import Network
from GeoLearning.Supervised import Regression
from Morphology.ShapeTools import Analyst
from Sandbox import proxy_indicators, proxy_network
from SB0_Variables import *
from UrbanZoning.City.Network import Streets
from matplotlib import rc
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from Visualization.Colors import get_rgb_values
from matplotlib import colors, cm

fm.fontManager.ttflist += fm.createFontList(['/Volumes/Samsung_T5/Fonts/avenir/Roboto-Light.ttf'])
rc('font', family='Roboto', weight='light')

MODES_COLORS = {'bus': cm.Blues, 'drive': cm.Reds, 'walk': cm.Greens, 'bike': cm.Oranges}
COLOR_MAP = {"drive": "D62728", "bus": "1F77B4", "walk": "2CA02C", "bike": "FF7F0E"}
MOB_MODES = list(MODES_COLORS.keys())

def del_obj_id(gdf):
    if 'OBJECTID' in gdf.columns:
        return gdf.drop('OBJECTID', axis=1)
    else:
        return gdf

def aerial_buffer(sample_gdf, layer_dir):
    """
    Aggregate data from layers to sample_gdf according to defined radii
    :param sample_gdf:
    :param layer_dir:
    :return:
    """
    # Test if sample_gdf has crs
    assert sample_gdf.crs is not None, AttributeError ('Assign coordinate reference system to sample_gdf')

    # Test if layers and columns exist within GeoPackage
    gdfs={layer: gpd.read_feather(f'{layer_dir}/{layer}.feather') for layer, cols in tqdm(network_layers.items())}
    for layer, cols in tqdm(network_layers.items()):
        for column in cols:
            assert column in gdfs[layer].columns, ValueError (f'{column} column not found in {layer} layer')

    for layer, cols in tqdm(network_layers.items()):
        for column in cols:
            right_gdf = gdfs[layer]

            # Test if right_gdf has crs
            assert right_gdf.crs is not None, AttributeError (f'Assign coordinate reference system to {layer}')

            # Test if column type is categorical or numerical
            if is_numeric_dtype(right_gdf[column]):
                for radius in radii:
                    sample_gdf = Analyst(sample_gdf).buffer_join(right_gdf.loc[:, [column, 'geometry']], radius)
            else:
                for category in right_gdf[column].unique():
                    filtered = right_gdf[right_gdf[column] == category]
                    filtered[category] = 1
                    for radius in radii:
                        sample_gdf = Analyst(sample_gdf).buffer_join(filtered.loc[:, [category, 'geometry']], radius)
    return sample_gdf

def analyze_sandbox(buildings, parcels, streets, sample_gdf=None, export=True, ch_dir=os.getcwd(), sb_name='Sandbox', suffix=''):
    # print(f"Analyzing {experiments} sandboxes")
    # for sandbox, value in experiments.items():
    #     proxy = Network(f'{sandbox}', crs=26910, directory=f'{directory}Sandbox/{sandbox}', nodes='network_intersections')
    #     db_layers = listlayers(proxy.gpkg)
    #
    #     # Check if sandbox has links and intersections
    #     network = [f'network_links_{suffix}', 'network_intersections', 'land_municipal_boundary']
    #     for layer in network:
    #         if layer not in db_layers:
    #             raise AttributeError(f"{layer} not found in GeoPackage of {sandbox}")
    #
    #     for code, year in experiments[sandbox][1].items():
    #         # Check if experiment has parcels and buildings
    #         built = [f'land_parcels_{code}', f'fabric_buildings_{code}']
    #         for layer in built:
    #             if layer not in db_layers:
    #                 raise AttributeError(f"{layer} not found in GeoPackage of {sandbox}")

    g_types = sample_gdf.geom_type.unique()
    if ('LineString' in list(g_types)) or ('MultiLineString' in list(g_types)):
        sample_gdf['geometry'] = sample_gdf.buffer(3)

    # Standardize directory
    if ch_dir[-1:] == '/': ch_dir = ch_dir[:-1]
    else: ch_dir = ch_dir

    # Segmentize streets
    streets = Streets(streets).segmentize().drop('id', axis=1)

    # Define geographic boundary
    sandbox = sb_name
    if f'{sandbox}.gpkg' in os.listdir(ch_dir): os.remove(f'{ch_dir}/{sandbox}.gpkg')
    if f'{sandbox}.gpkg-journal' in os.listdir(ch_dir): os.remove(f'{ch_dir}/{sandbox}.gpkg-journal')
    proxy = Network(f'{sandbox}', crs=26910, directory=f'{ch_dir}', nodes='network_intersections')

    # Save parcels, buildings and streets on local_gbd GeoPackage
    time.sleep(1)
    del_obj_id(streets).to_file(proxy.gpkg, layer='network_links', driver='GPKG')
    time.sleep(1)
    del_obj_id(parcels).to_file(proxy.gpkg, layer=f'land_parcels_{suffix}', driver='GPKG')
    time.sleep(1)
    del_obj_id(buildings).to_file(proxy.gpkg, layer=f'fabric_buildings_{suffix}', driver='GPKG')

    # Transfer network indicators to sandbox
    proxy = proxy_network(proxy)

    # Extract elevation data
    proxy.node_elevation(elevation=False)

    # Calculate spatial indicators
    if 'E0' in suffix: year = 2020
    else: year = 2040
    proxy = proxy_indicators(proxy, experiment={suffix: year})

    # Perform network buffer analysis
    if sample_gdf is None: sample_gdf = gpd.read_file(proxy.gpkg, layer=f"land_parcels_{suffix}")

    # # Perform aerial buffer analysis
    # sample_gdf = aerial_buffer(sample_gdf, proxy.gpkg)

    sample_gdf = proxy.network_analysis(
        run=True,
        col_prefix='mob',
        file_prefix=f'mob_{suffix}',
        service_areas=radii,
        sample_gdf=sample_gdf,
        aggregated_layers=network_layers,
        keep=['OBJECTID', "population, 2016"],
        export=export)

    # # Divide sums aggregations to a buffer overlay in order to avoid edge effects
    # for col in results.columns:
    #     if '_sum_' in col:
    #         results[col] = results[col]/results['divider']

    gc.collect()
    return sample_gdf

def rename_features(rename):
    rename_dict2 = {}
    rename_dict3 = {}
    type_dict2 = {}

    for k, value in rename.items():
        v = value[0]
        rename_dict3[k] = v
        for r in radii:
            for d in ['l', 'f']:
                for op in ['ave', 'sum', 'cnt', 'rng']:
                    if op == 'ave':
                        t = 'Average'
                    elif op == 'sum':
                        t = 'Total'
                    else:
                        t = ''
                    renamed = f'{t} '
                    item = f'{renamed}{v.lower()} within {r}m'
                    rename_dict2[f"{k}_r{r}_{op}_{d}"] = f'{item.strip()[0].upper()}{item.strip()[1:]}'
                    type_dict2[f"{k}_r{r}_{op}_{d}"] = value[1]
    rename = {**rename_dict2, **rename_dict3}
    return rename

def train_regression(training, label_cols, rename=None, columns=None, suffix='', ch_dir='', random_seeds=6):
    # proxy_files2 = list(testing.values())

    if rename is not None: rename = rename_features(rename)
    else: rename = {}
    if columns is None: columns = training[0].columns

    for rs in range(random_seeds):
        # # Filter columns common to proxy and dissemination areas
        # ind_cols = [set(gdf.columns) for gdf in training]+[set(columns)]
        # common_cols = list(set.intersection(*ind_cols))
        # final_cols = [col for col in common_cols if col in list(rename.keys())]
        # training = [gdf.loc[:, final_cols + label_cols] for gdf in training]

        print(f"\nStarting regression with random seed {rs}")
        reg = Regression(
            r_seed=rs,
            test_split=0.2,
            n_indicators=5,
            round_f=4,
            norm_x=False,
            norm_y=False,
            data=training,
            directory=ch_dir,
            predicted=label_cols,
            prefix=f'',
            rename=rename,
            filter_pv=False,
            plot=True,
            pv=0.05,
            file_suffix=f"{suffix}_{rs}",
            color_by="Population density per square kilometre, 2016",
        )

        # Run random forest and partial dependence plots
        reg.non_linear(method=RandomForestRegressor)
        reg.test_non_linear(i_method='regular')

        features = reg.partial_dependence(n_features=9)
        reg.save_model()
    return reg

def test_regression(proxy_gdf, label_cols, random_seeds=6, ch_dir=os.getcwd(), suffix=''):

    # Create regression object
    reg = Regression(
        test_split=0.2,
        n_indicators=5,
        round_f=4,
        norm_x=False,
        norm_y=False,
        directory=ch_dir,
        predicted=label_cols,
        prefix=f'',
        filter_pv=False,
        plot=True,
        pv=0.05,
        color_by="Population density per square kilometre, 2016",
    )

    # Iterate over proxy files
    all_seeds = proxy_gdf.copy()
    for mode in reg.label_cols:
        for rs in range(random_seeds):
            print(f"\nStarting regression with random seed {rs}")
            reg.r_seed = rs
            reg.fitted = pickle.load(open(f'/Volumes/Samsung_T5/Databases/Regressed/Regression/{mode}_FittedModel_{rs}.sav', 'rb'))
            reg.train_data = pickle.load(open(f'/Volumes/Samsung_T5/Databases/Regressed/Regression/{mode}_TrainData_{rs}.sav', 'rb'))

            # Predict sandbox using random forest
            proxy_gdf_rs = reg.pre_norm_exp(proxy_gdf, normalize=True, prefix=f'rf_{rs}')
            gc.collect()

            # Append prediction to all_seeds
            all_seeds = pd.concat([all_seeds, proxy_gdf_rs.loc[:, [f'{mode}_rf_{rs}_n']]], axis=1)

        # Get most important features

    # Average random seeds
    for label in label_cols:
        all_seeds[label] = all_seeds.loc[:, [f'{label}_rf_{rs}_n' for rs in range(random_seeds)]].mean(axis=1)

    # Return parcels with predicted mode shares
    gc.collect()
    return all_seeds

def dict_to_pandas(dictionary):
    dfs = []
    for k in dictionary.keys():
        df = dictionary[k]
        df['key'] = k
        dfs.append(df)
    return pd.concat(dfs)

def load_importance(random_seeds=6):
    all_df = pd.DataFrame()
    for rs in range(random_seeds):
        imp_dict = pickle.load(open(f'{directory}/Sandbox/Sunset/Regression/ImportanceData_Sunset_{rs}.sav', 'rb'))
        df = dict_to_pandas(imp_dict)
        df['rs'] = rs
        all_df = pd.concat([all_df, df])
    all_df = all_df.sort_values('importance', ascending=False)
    return all_df

def predict_mobility(gdf, mob_modes):
    predicted = test_regression(proxy_gdf=gdf, label_cols=mob_modes, random_seeds=r_seeds).to_crs(4326)  # .loc[:, mob_modes + ['geometry']].to_crs(4326)
    return predicted.loc[:,~predicted.columns.duplicated()]

def build_choropleth_map(gdf, mode, layer):

    predicted = gdf.to_crs(26910)
    predicted = predicted.to_crs(4326)

    predicted['active'] = predicted['walk'] + predicted['bike']
    predicted = get_rgb_values(predicted, mode, MODES_COLORS[mode])

    layer.streets = predicted
    layer.export_choropleth_pdk(mode, layer='streets', color_map=MODES_COLORS[mode], n_breaks=100)
    return

def build_pie_chart(gdf):
    predicted = gdf
    # Create pie chart with mode shares
    shares_df = pd.DataFrame()
    shares_df['names'] = [mode.title() for mode in MOB_MODES]
    shares_df['shares'] = list(predicted.loc[:, MOB_MODES].mean())
    shares = px.pie(shares_df, names='names', values='shares', color='names', hole=0.6, opacity=0.8,
                    color_discrete_map={k.title(): v for k, v in COLOR_MAP.items()})
    shares.update_layout(margin={'b': 0, 'l': 0, 'r': 0, 't': 0}, legend=dict(orientation="h"), paper_bgcolor='rgba(0,0,0,0)',
                         plot_bgcolor='rgba(0,0,0,0)')

    return shares

def build_histogram(gdf):
    predicted = gdf
    # Build histogram
    disaggregated = pd.DataFrame()
    prd_copy = predicted.copy()
    for mode in MOB_MODES:
        prd_copy['mode'] = mode
        prd_copy['share'] = prd_copy[mode]
        disaggregated = pd.concat([disaggregated, prd_copy])

    color_discrete_map = {mode: f"#{COLOR_MAP[mode]}" for mode in MOB_MODES}
    hist = px.histogram(disaggregated, x='share', color='mode', color_discrete_map=color_discrete_map)
    hist.update_layout(margin={'b': 0, 'l': 0, 'r': 0, 't': 0}, paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    hist.update_traces(opacity=0.6)
    return hist


directory = '/Volumes/Samsung_T5/Databases'
rename_dict = {
    'mob_network_stops_ct': ('Public transit stops', 'network'),
    'mob_frequency': ('Transit frequency', 'network'),
    # 'mob_network_nodes_ct': ('Number of intersections', 'network'),
    # 'mob_elevation': ('Elevation', 'network'),

    # 'mob_connectivity': ('Axial connectivity', 'network'),
    # 'mob_axial_closeness': ('Axial closeness centrality', 'network'),
    # 'mob_axial_betweenness': ('Axial betweenness centrality', 'network'),
    # 'mob_axial_n_betweenness': ('Normalized axial betweenness centrality', 'network'),
    # 'mob_axial_length': ('Axial line length', 'network'),
    # 'mob_axial_eigenvector': ('Axial eigenvector centrality', 'network'),
    # 'mob_axial_katz': ('Axial katz centrality', 'network'),
    # 'mob_axial_pagerank': ('Axial page rank centrality', 'network'),
    # 'mob_axial_hits1': ('Axial hits centrality', 'network'),
    # 'mob_axial_degree': ('Axial degree centrality', 'network'),

    'mob_network_walk_ct': ('Intensity of walkable Network', 'network'),
    'mob_network_bike_ct': ('Intensity of bikeable Network', 'network'),
    'mob_network_drive_ct': ('Intensity of driveable Network', 'network'),
    'mob_walk_length': ('Length of walkable Network', 'network'),
    'mob_bike_length': ('Length of bikeable Network', 'network'),
    'mob_drive_length': ('Length of driveable Network', 'network'),
    # 'mob_walk_straight': ('Straightness of walkable Network', 'network'),
    # 'mob_bike_straight': ('Straightness of bikeable Network', 'network'),
    # 'mob_drive_straight': ('Straightness of driveable Network', 'network'),

    'mob_land_assessment_fabric_ct': ('Number of units', 'density'),
    'mob_n_use': ('Use diversity', 'landuse'),
    'mob_CM': ('Commercial', 'landuse'),
    'mob_SFD': ('Single-Family Detached', 'landuse'),
    'mob_SFA': ('Single-Family Attached', 'landuse'),
    'mob_MFL': ('Multi-Family Low-Rise', 'landuse'),
    'mob_MFH': ('Multi-Family High-Rise', 'landuse'),
    'mob_MX': ('Mixed Use', 'landuse'),
    'mob_total_finished_area': ('Total finished area', 'density'),
    'mob_gross_building_area': ('Gross building area', 'density'),
    'mob_number_of_bedrooms': ('Number of bedrooms', 'density'),

    # 'mob_land_assessment_parcels_ct': ('Number of parcels', 'density'),
    # 'mob_area_sqkm': ('Parcel size', 'density'),
    # 'mob_n_size': ('Parcel diversity', 'density'),

    'mob_population density per square kilometre, 2016': ('Population density', 'density'),
    'mob_n_dwellings': ('Number of dwellings', 'density'),
    'mob_population, 2016': ('Population', 'density'),

}
infra_dict = {
    'bus': {
        'mob_frequency': ('Transit frequency', 'network'),
        'mob_network_stops_ct': ('Public transit stops', 'network'),
    },
    'bike': {
        'mob_cycle_length': ('Cycling network length', 'network'),
    }
}

if __name__ == '__main__':
    drop = [
        'dauid',
        'pruid',
        'prname',
        'cduid',
        'cdname',
        'cdtype',
        'ccsuid',
        'ccsname',
        'csduid',
        'csdname',
        'csdtype',
        'eruid',
        'ername',
        'saccode',
        'sactype',
        'cmauid',
        'cmapuid',
        'cmaname',
        'cmatype',
        'ctuid',
        'ctname',
        'adauid',
        'geographic code',
        'province / territory, english',
        'province / territory, french',
        'geographic code, province / territory',
        'geographic code, census division',
        'geographic code, census subdivision',
        'population, 2016',
        'incompletely enumerated indian reserves and indian settlements, 2016',
        'n_dwellings',
        'private dwellings occupied by usual residents, 2016',
        'land area in square kilometres, 2016',
        'population density per square kilometre, 2016',
        'unnamed: 12',
        'index_right',
        'place_name',
        'bbox_north',
        'bbox_south',
        'bbox_east',
        'bbox_west',
        'age_div_sh',
        'ethnic_div_sh',
        'ethnic_div_si',
        'ed_total',
        'no_certificate',
        'secondary',
        'postsecondary',
        'no_postsecondary',
        'education',
        'arts',
        'humanities',
        'social',
        'business',
        'natural',
        'information',
        'engineering',
        'agriculture',
        'health',
        'protective',
        'other',
        'no_certificate_ratio',
        'secondary_ratio',
        'postsecondary_ratio',
        'no_postsecondary_ratio',
        'educ_div_sh',
        'educ_div_si',
        'total_tenure',
        'owner',
        'renter',
        'band housing',
        'total_cond',
        'condominium',
        'not condominium',
        'total_bedr',
        'no bedrooms',
        '1 bedroom',
        '2 bedrooms',
        '3 bedrooms',
        '4 or more bedrooms',
        'total_rooms',
        '1 to 4 rooms',
        '5 rooms',
        '6 rooms',
        '7 rooms',
        '8 or more rooms',
        'ave_n_rooms',
        'total_people_per_room',
        'one person or fewer per room',
        'more than 1 person per room',
        'total_suitability',
        'suitable',
        'not suitable',
        'total_period',
        '1960 or before',
        '1961 to 1980',
        '1981 to 1990',
        '1991 to 2000',
        '2001 to 2005',
        '2006 to 2010',
        '2011 to 2016',
        'owned_med_cost',
        'owned_ave_cost',
        'owned_med_dwe_value',
        'owned_ave_dwe_value',
        'total_tenant',
        'receives_subsidy_rat',
        'more30%income_rat',
        'rented_med_cost',
        'rented_ave_cost',
        'owner_ratio',
        'renter_ratio',
        'condominium_ratio',
        'not_condominium_ratio',
        'no_bedrooms_ratio',
        '1_bedroom_ratio',
        '2_bedrooms_ratio',
        '3_bedrooms_ratio',
        '4_plus_bedrooms_ratio',
        '1_4_rooms_ratio',
        '5_rooms_ratio',
        '6_rooms_ratio',
        '7_rooms_ratio',
        '8_plus_rooms_ratio',
        '1_person_per_room_ratio',
        '1_plus_person_per_room_ratio',
        'suitable_ratio',
        'not_suitable_ratio',
        'building_age_div_si',
        'building_age_div_sh',
        'dwelling_div_rooms_si',
        'dwelling_div_rooms_sh',
        'dwelling_div_bedrooms_si',
        'dwelling_div_bedrooms_sh',
        'income_recipients',
        'median_income',
        'after_tax_recipients',
        'med_income_after_tax',
        'mkt_recipients',
        'med_mkt_income',
        'n_gov_transfers',
        'median_gov_transfers',
        'n_emp_income',
        'med_emp_income',
        'total_income_stats',
        'mkt_income',
        'emp_income',
        'gov_transfers',
        'total_income_groups',
        'without_income',
        'with_income',
        'income_ratio',
        'total_income_after_tax',
        'without_income_at',
        'with_income_at',
        'income_ratio_at',
        '    under $10;000 (including loss)',
        '    $10;000 to $19;999',
        '    $20;000 to $29;999',
        '    $30;000 to $39;999',
        '    $40;000 to $49;999',
        '    $50;000 to $59;999',
        '    $60;000 to $69;999',
        '    $70;000 to $79;999',
        '    $80;000 and over',
        '      $80;000 to $89;999',
        '      $90;000 to $99;999',
        '      $100;000 and over',
        'income_div_si',
        'income_div_sh',
        'total - main mode of commuting for the employed labour force aged 15 years and over in private households with a usual place of work or no fixed workplace address - 25% sample data',
        '  car; truck; van - as a driver',
        '  car; truck; van - as a passenger',
        '  public transit',
        '  walked',
        '  bicycle',
        '  other method',
        'total_labour_force',
        'in the labour force',
        'employed',
        'unemployed',
        'not in the labour force',
        'participation rate',
        'employment rate',
        'unemployment rate',
        'total_class_worker',
        'class of worker - not applicable',
        'all classes of workers',
        'employee',
        'self-employed',
        'total_place',
        'worked at home',
        'worked outside canada',
        'no fixed workplace address',
        'worked at usual place',
        'worked_home_ratio',
        'worked_abroad_ratio',
        'worked_flexible',
        'worked_usual_ratio',
    ]
    columns = [
        'mob_CM_r1200_sum_f',
        'mob_number_of_bedrooms_r1200_sum_f',
        'mob_frequency_r1200_ave_f',
        'mob_n_dwellings_r1200_sum_f',
        'mob_population, 2016_r400_ave_f',
        'mob_MFL_r1200_ave_f',
        'mob_bike_length_r1200_sum_f',
        'mob_population density per square kilometre, 2016_r1200_sum_f',
        'mob_walk_length_r1200_sum_f',
        'walk',
        'drive',
        'bike',
        'bus'
    ]
    train_regression(label_cols=['walk', 'drive', 'bike', 'bus'], ch_dir='/Volumes/Samsung_T5/Databases/Regressed', training=[pd.concat([
        gpd.read_feather('/Volumes/Samsung_T5/Databases/Network/Metro Vancouver, British Columbia_mob_na.feather').loc[:, columns],
        # gpd.read_feather('/Volumes/Samsung_T5/Databases/Network/Capital Regional District, British Columbia_mob_na.feather').loc[:, columns]
    ])])
