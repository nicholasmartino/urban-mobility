from Geospatial import GeoBoundary, BritishColumbia, Canada
import geopandas as gpd

# Process mobility related indicators for elementslab
regions = {
    'Canada':
        {
            'British Columbia': ['Capital Regional District', 'Metro Vancouver', 'Prince George']
        }
}

for key, value in regions.items():
    bc = BritishColumbia(cities=value['British Columbia'])
    for city in bc.cities: city.update_databases(bound=True, net=True)

    bca_dir = '/Volumes/ELabs/50_projects/16_PICS/07_BCA data/'
    bc.aggregate_bca_from_field(
        inventory_dir=f'{bca_dir}170811_BCA_Provincial_Data/Inventory Information - RY 2017.csv',
        geodatabase_dir=f'{bca_dir}Juchan_backup/BCA_2017_roll_number_method/BCA_2017_roll_number_method.gdb')

    for city in bc.cities:
        city.network_analysis(
            service_areas=[400, 800, 1600],
            sample_gdf=gpd.read_file(city.gpkg, layer='land_dissemination_area'),
            aggregated_layers={
                'network_links': ["network_links_ct"],
                'network_nodes': ["network_nodes_ct"],
                'land_assessment_fabric': [
                    "land_assessment_fabric_ct", "NUMBER_OF_BEDROOMS", "NUMBER_OF_BATHROOMS", "elab_use"],
                'land_parcels': ["land_parcels_ct", "elab_size"]
            })

    city.set_parameters(unit='lda', service_areas=[400, 800, 1600], samples=None)
    # city.regression()
    # city.export_map()
    # city.network_analysis(sample='lda', service_areas=[400, 800, 1600], cols={})
    # city.density_indicators()
    # city.diversity_indicators()
    # city.street_network_indicators()
    # city.cycling_network_indicators()
    # city.export_databases()
    # city.export_destinations()
    # sandbox = Sandbox(name=value[1], geodatabase=value[2], layers=value[3])
    # sandbox.morph_indicators()

parcels = {
    "BC": "https://pub.data.gov.bc.ca/datasets/4cf233c2-f020-4f7a-9b87-1923252fbc24/ParcelMapBCExtract.zip"
}
