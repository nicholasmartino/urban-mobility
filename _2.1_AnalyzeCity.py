import gc

import geopandas as gpd
from UrbanScraper.Local import BritishColumbia
from UrbanScraper.Scraper import Canada
from _0_Variables import *

# Perform network analysis for the real place
for key, value in regions.items():
    bc = BritishColumbia(cities=value['British Columbia'], directory=directory)
    country = Canada(provinces=[bc])

    for city in bc.cities:
        city.update_databases(net=False)
        city.centrality(run=True, axial=True, layer='network_walk')
        city.centrality(run=False, osm=True, layer='network_drive')
        city.node_elevation(run=True)
        network_analysis = city.network_analysis(
            prefix='mob',
            run=False,
            service_areas=radius,
            sample_gdf=gpd.read_file(city.gpkg, layer='land_dissemination_area'),
            decays=['flat'],
            filter_min={'population density per square kilometre, 2016': 300},
            aggregated_layers=network_layers,
            keep=['dauid', 'walk', 'bike', 'drive', 'bus', 'geometry'])
        gc.collect()
