import geopandas as gpd
from Analyst import Network
from Sandbox import proxy_indicators, proxy_network, overlay_radius
from SB0_Variables import directory, radii, network_layers
from fiona import listlayers


def analyze_sandbox(experiments, district=True, export=True):
    print(f"Analyzing {experiments} sandboxes")
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

    # Perform network analysis
    na = {}
    for sandbox, value in experiments.items():
        na[f"{sandbox}"] = {}

        # Define geographic boundary
        proxy = Network(f'{sandbox} Sandbox', crs=26910, directory=f'{directory}Sandbox/{sandbox}', nodes='network_intersections')

        # Transfer network indicators to sandbox
        proxy = proxy_network(proxy)

        # Extract elevation data
        proxy.node_elevation()

        for code, year in experiments[sandbox][1].items():
            if district:
                district_net = Network(experiments[sandbox][0], crs=26910)
                for layer in ['network_axial']:
                    overlay_radius(proxy.gpkg, district_net.gpkg, sample_layer=layer)

            # Calculate spatial indicators
            proxy = proxy_indicators(proxy, experiment={code: year})

            # Perform network analysis
            results = proxy.network_analysis(
                run=True,
                col_prefix='mob',
                file_prefix=f'mob_{code}',
                service_areas=radii,
                sample_gdf=gpd.read_file(proxy.gpkg, layer=f"land_parcels_{code}"),
                aggregated_layers=network_layers,
                keep=['OBJECTID', "population, 2016"],
                export=export)

            # Divide sums aggregations to a buffer overlay in order to avoid edge effects
            for col in results.columns:
                if '_sum_' in col:
                    results[col] = results[col]/results['divider']

            na[f"{sandbox}"][f"{code}"] = results
    return na
