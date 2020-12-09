import geopandas as gpd
from Analyst import Network
from Sandbox import proxy_indicators, proxy_network
from _0_Variables import *
from fiona import listlayers


for sandbox, value in experiments.items():
    proxy = Network(f'{sandbox} Sandbox', crs=26910, directory=f'{directory}Sandbox/{sandbox}', nodes='network_intersections')
    db_layers = listlayers(proxy.gpkg)

    # Check if sandbox has links and intersections
    network = ['network_links', 'network_intersections', 'land_municipal_boundary']
    for layer in network:
        if layer not in db_layers:
            raise AttributeError(f"{layer} not found in GeoPackage of {sandbox}")

    for code, year in experiments[sandbox][1].items():
        # Check if experiment has parcels and buildings
        built = [f'land_parcels_{code}', f'fabric_buildings_{code}']
        for layer in built:
            if layer not in db_layers:
                raise AttributeError(f"{layer} not found in GeoPackage of {sandbox}")


# # Check if all variables exist in database
# def check_layers(all_layers, geopackage):
#     db_layers = listlayers(geopackage)
#
#     for layer in all_layers.keys():
#         if layer in db_layers:
#             layer_df = gpd.read_file(proxy.gpkg, layer=layer)
#             difference = set(all_layers[layer]).difference(layer_df.columns)
#             if len(difference) > 0:
#                 raise AttributeError(f"{difference} not found in {layer} layer")
#         else:
#             raise AttributeError(f"{layer} not found in GeoPackage")
#
# for sandbox, value in experiments.items():
#     proxy = Network(f'{sandbox} Sandbox', crs=26910, directory=f'{directory}Sandbox/{sandbox}', nodes='network_intersections')
#
#     # Check if network layers are within GeoPackage
#     check_layers(network_layers, proxy.gpkg)
#     check_layers(network_bike, proxy.gpkg)
#     check_layers(network_bus, proxy.gpkg)


# Perform network analysis

for sandbox, value in experiments.items():

    # Define geographic boundary
    proxy = Network(f'{sandbox} Sandbox', crs=26910, directory=f'{directory}Sandbox/{sandbox}', nodes='network_intersections')

    # Transfer network indicators to sandbox
    proxy = proxy_network(proxy)

    # Extract elevation data
    proxy.node_elevation()

    for code, year in experiments[sandbox][1].items():
        district = Network(experiments[sandbox][0], crs=26910)

        # Calculate spatial indicators
        proxy = proxy_indicators(proxy, district, experiment={code: year})

        # Calculate spatial indicators
        p_gdf = gpd.read_file(proxy.gpkg, layer=f'land_parcels_{code}')

        # Perform network analysis
        network_analysis = proxy.network_analysis(
            run=True,
            col_prefix='mob',
            file_prefix=f'mob_{code}',
            service_areas=radius,
            sample_gdf=gpd.read_file(proxy.gpkg, layer=f"land_parcels_{code}"),
            aggregated_layers=network_layers,
            keep=['OBJECTID', "population, 2016"])
