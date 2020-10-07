import gc

import geopandas as gpd
from Analyst import GeoBoundary
from Geospatial.Scraper import BritishColumbia, Canada
from Sandbox import proxy_indicators, proxy_network
from _0_Variables import regions, radius, network_layers, network_bike, network_bus


# Perform same analysis with sandbox (proxy)
hq_experiments = {
    'e0':2020,
    'e1':2040,
    'e2':2040,
    'e3':2040
}
ss_experiments = {
    'e0':2020,
    'e1':2030,
    'e2':2030,
    'e3':2030,
    'e4':2040,
    'e5':2040,
    'e6':2040,
    'e7':2050,
    'e8':2050
}
experiments={
    #'Sunset': ['Metro Vancouver, British Columbia', ss_experiments],
    'Hillside Quadra': ['Capital Regional District, British Columbia', hq_experiments]
}

# Perform network analysis for the real place
for key, value in regions.items():
    bc = BritishColumbia(cities=value['British Columbia'])
    country = Canada(provinces=[bc])

    for city in bc.cities:
        city.centrality(run=False, axial=True, layer='network_walk')
        city.centrality(run=False, osm=True, layer='network_drive')
        city.node_elevation(run=False)
        filter_min = {'population density per square kilometre, 2016': 300}
        network_analysis = city.network_analysis(
            prefix='mob',
            run=False,
            service_areas=radius,
            sample_layer='land_dissemination_area',
            decays=['flat'],
            filter_min=filter_min,
            aggregated_layers=network_layers,
            keep=['dauid', 'walk', 'bike', 'drive', 'bus', 'geometry'])
        gc.collect()

# Read sandbox
for sandbox, value in experiments.items():

    # Define geographic boundary
    proxy = GeoBoundary(f'{sandbox} Sandbox', crs=26910, directory=f'/Volumes/Samsung_T5/Databases/Sandbox/{sandbox}')

    # Transfer network indicators to sandbox
    proxy = proxy_network(proxy)
    proxy.nodes = gpd.read_file(proxy.gpkg, layer='network_intersections')
    proxy.links = gpd.read_file(proxy.gpkg, layer='network_links')

    # Extract elevation data
    proxy.node_elevation()

    for code, year in experiments[sandbox][1].items():
        district = GeoBoundary(experiments[sandbox][0], crs=26910)

        # Calculate spatial indicators
        proxy = proxy_indicators(proxy, district, experiment={code: year})
        p_gdf = gpd.read_file(proxy.gpkg, layer=f'land_parcels_{code}')

        # Perform network analysis
        network_analysis = proxy.network_analysis(
            run=False,
            col_prefix='mob',
            file_prefix=f'mob_{code}',
            service_areas=radius,
            sample_layer=f"land_parcels_{code}",
            aggregated_layers=network_layers,
            keep=['OBJECTID', "population, 2016"])

        # Perform network analysis
        network_analysis_bus = proxy.network_analysis(
            run=False,
            col_prefix='mob_bus',
            file_prefix=f'mob_bus_{code}',
            service_areas=radius,
            sample_layer=f"land_parcels_{code}",
            aggregated_layers=network_bus,
            keep=['OBJECTID', "population, 2016"])

        # Perform network analysis
        network_analysis_bike = proxy.network_analysis(
            run=True,
            col_prefix='mob_bike',
            file_prefix=f'mob_bike_{code}',
            service_areas=radius,
            sample_layer=f"land_parcels_{code}",
            aggregated_layers=network_bike,
            keep=['OBJECTID', "population, 2016"])
