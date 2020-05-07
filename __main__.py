import datetime

import geopandas as gpd
from Geospatial.Scraper import GeoScraper, BritishColumbia

# Process mobility related indicators for elementslab
regions = {
    'Canada':
        {
            'British Columbia': ['Capital Regional District']
        }
}

for key, value in regions.items():
    date_list = [str((datetime.datetime(2016, 5, 31) - datetime.timedelta(days=x)).date()) for x in range(30)]

    bc = BritishColumbia(cities=value['British Columbia'])
    for city in bc.cities: city.update_databases(bound=False, net=False)

    windows = False
    if windows: bca_dir = '//nas.sala.ubc.ca/ELabs/50_projects/16_PICS/07_BCA data/'
    else: bca_dir = '/Volumes/ELabs/50_projects/16_PICS/07_BCA data/'

    bc.aggregate_bca_from_field(
        run=False,
        inventory_dir=f'{bca_dir}170811_BCA_Provincial_Data/Inventory Information - RY 2017.csv',
        geodatabase_dir=f'{bca_dir}Juchan_backup/BCA_2017_roll_number_method/BCA_2017_roll_number_method.gdb')

    for city in bc.cities:
        scraper = GeoScraper(city=city)
        for date in date_list: scraper.public_transit(True, date)  # Transit Land
        city.centrality(run=True)
        city.network_analysis(
            service_areas=[400, 800, 1600],
            sample_gdf=gpd.read_file(city.gpkg, layer='land_dissemination_area'),
            aggregated_layers={
                'network_links': ["network_links_ct", "betweenness", "n_betweenness"],
                'network_nodes': ["network_nodes_ct", "closeness", "betweenness", "n_betweenness"],
                'land_assessment_fabric': [
                    "land_assessment_fabric_ct", "NUMBER_OF_BEDROOMS", "NUMBER_OF_BATHROOMS", "elab_use"],
                'land_assessment_parcels': ["land_assessment_parcels_ct", "elab_size"]
            })

parcels = {
    "BC": "https://pub.data.gov.bc.ca/datasets/4cf233c2-f020-4f7a-9b87-1923252fbc24/ParcelMapBCExtract.zip"
}
