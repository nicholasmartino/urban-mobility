import datetime
import gc
import pandas as pd
import geopandas as gpd
from Geospatial.Scraper import GeoScraper, BritishColumbia, Canada
from Analyst import filter_features

regions = {
    'Canada':
        {
            'British Columbia': [
                #'Capital Regional District',
                'Metro Vancouver',
                #'Hillside Quadra Proxy'
            ]
        }
}

network_layers = {
    'network_nodes': ["elevation", "node_closeness", "node_betweenness", "node_n_betweenness"],
    'network_axial': ["axial_closeness", "axial_betweenness", "axial_length"],
    'network_drive': ["link_betweenness", "link_n_betweenness"],
    'network_cycle': ["cycle_length"],

    'land_assessment_fabric': ["n_use", "year_built", "total_finished_area", "gross_building_area",
        "number_of_bedrooms", "number_of_bathrooms"],
    'land_assessment_parcels': ["area_sqkm", "n_size"],
    'land_dissemination_area': ["population, 2016", "population density per square kilometre, 2016",
        "n_dwellings", "building_age_div_si", "building_age_div_sh", "dwelling_div_rooms_si",
        "dwelling_div_bedrooms_si", "dwelling_div_rooms_sh", "dwelling_div_bedrooms_sh", ]
}

radius = [400, 800, 1600, 3200]

for key, value in regions.items():
    bc = BritishColumbia(cities=value['British Columbia'])
    country = Canada(provinces=[bc])

    # StatsCan
    country.update_databases(census=False)

    # OpenStreetMaps
    for city in bc.cities: city.update_databases(bound=False, net=False)

    # BC Assessment
    windows = False
    if windows: bca_dir = '//nas.sala.ubc.ca/ELabs/50_projects/16_PICS/07_BCA data/'
    else: bca_dir = '/Volumes/ELabs/50_projects/16_PICS/07_BCA data/'
    bc.aggregate_bca_from_field(
        run=False, join=True, classify=True,
        inventory_dir=f'{bca_dir}170811_BCA_Provincial_Data/Inventory Information - RY 2017.csv',
        geodatabase_dir=f'{bca_dir}Juchan_backup/BCA_2017_roll_number_method/BCA_2017_roll_number_method.gdb')

    # BC Transit
    bc.get_bc_transit(run=False, down=False, urls=[
        #'http://victoria.mapstrat.com/current/google_transit.zip',
        'local',
        #'https://www.bctransit.com/data/gtfs/prince-george.zip'
    ])

    for city in bc.cities:
        scraper = GeoScraper(city=city)
        city.centrality(run=False, axial=True, layer='network_walk')
        city.centrality(run=False, osm=True, layer='network_drive')
        city.node_elevation(run=True)
        filter_min = {'population density per square kilometre, 2016': 300}
        network_analysis = city.network_analysis(
            run=True,
            service_areas=radius,
            sample_layer='land_dissemination_area',
            decays=['linear'],
            filter_min=filter_min,
            aggregated_layers=network_layers,
            keep=['dauid', 'walk', 'bike', 'drive', 'bus', 'geometry'])
        gc.collect()

    x_features = list(gpd.read_file(bc.cities[0].gpkg, layer='regression_features')['features'])
    print(f"Joined {len(x_features)} features")

    filtered = filter_features(
        df=pd.concat([gpd.read_file(city.gpkg, layer='network_analysis_') for city in bc.cities]),
        x_features=x_features,
        y_features=['walk', 'bike', 'drive', 'bus'],
        pval_threshold=0.05,
        corr_threshold=0.1
    )

    # Export analyzed features
    filtered['feature'] = filtered.index
    filtered = filtered.reset_index(drop=True)
    filtered.to_csv(f'{bc.cities[0].directory}FeaturesNN.csv')

    # # Perform same analysis with proxy
    # proxy = GeoBoundary('Hillside Quadra Proxy')
    # p_gdf = gpd.read_file(proxy.gpkg, layer='land_parcels')
    # network_analysis = proxy.network_analysis(
    #     run=True,
    #     service_areas=radius,
    #     sample_gdf=p_gdf,
    #     aggregated_layers=network_layers)

parcels = {"BC": "https://pub.data.gov.bc.ca/datasets/4cf233c2-f020-4f7a-9b87-1923252fbc24/ParcelMapBCExtract.zip"}
