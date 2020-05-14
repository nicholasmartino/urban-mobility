import datetime

import geopandas as gpd
from Geospatial.Scraper import GeoScraper, BritishColumbia, Canada

# Process mobility related indicators for elementslab
regions = {
    'Canada':
        {
            'British Columbia': ['Capital Regional District', 'Metro Vancouver']
        }
}

for key, value in regions.items():
    date_list = [str((datetime.datetime(2018, 5, 31) - datetime.timedelta(days=x)).date()) for x in range(30)]

    bc = BritishColumbia(cities=value['British Columbia'])
    country = Canada(provinces=[bc])

    country.update_databases(census=False)  # StatsCan
    for city in bc.cities: city.update_databases(bound=False, net=False)

    windows = False
    if windows: bca_dir = '//nas.sala.ubc.ca/ELabs/50_projects/16_PICS/07_BCA data/'
    else: bca_dir = '/Volumes/ELabs/50_projects/16_PICS/07_BCA data/'

    bc.get_bc_transit(urls=['http://victoria.mapstrat.com/current/google_transit.zip',
                            'local'], run=False, down=False)  # BC Transit
    bc.aggregate_bca_from_field(
        run=False, join=False, classify=False,
        inventory_dir=f'{bca_dir}170811_BCA_Provincial_Data/Inventory Information - RY 2017.csv',
        geodatabase_dir=f'{bca_dir}Juchan_backup/BCA_2017_roll_number_method/BCA_2017_roll_number_method.gdb')

    for city in bc.cities:
        scraper = GeoScraper(city=city)
        for date in date_list: scraper.public_transit(False, date)  # Transit Land
        lda_gdf = gpd.read_file(city.gpkg, layer='land_dissemination_area')
        city.centrality(run=True, axial=True, layer='network_walk')
        city.centrality(run=True, osm=True, layer='network_drive')
        x_features = city.network_analysis(
            run=True,
            service_areas=[400, 800, 1600],
            sample_gdf=lda_gdf,
            aggregated_layers={
                'network_axial': ["closeness", "betweenness", "length"],
                'network_nodes': ["closeness", "betweenness"],
                'network_cycle': ['length'],
                'network_drive': ["betweenness"],
                'network_transit': ['frequency'],
                'land_assessment_fabric': ["n_use", "YEAR_BUILT", "TOTAL_FINISHED_AREA", "GROSS_BUILDING_AREA",
                                           "NUMBER_OF_BEDROOMS", "NUMBER_OF_BATHROOMS"],
                'land_assessment_parcels': ['area_sqkm', 'n_size'],
                'land_dissemination_area': ['Population, 2016', 'Population density per square kilometre, 2016',
                                            'Total private dwellings, 2016']
            })
        city.p_values(
            df=lda_gdf,
            x_features=x_features,
            y_features=['walk', 'bike', 'drive', 'bus'])

parcels = {
    "BC": "https://pub.data.gov.bc.ca/datasets/4cf233c2-f020-4f7a-9b87-1923252fbc24/ParcelMapBCExtract.zip"
}
