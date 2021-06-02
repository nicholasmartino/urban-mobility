import gc

import geopandas as gpd
from UrbanScraper.Local import BritishColumbia
from UrbanScraper.Scraper import Canada
from SB0_Variables import *
from UrbanMobility.DashPredict_Back import aerial_buffer

# Perform network analysis for the real place
for key, value in regions.items():
    bc = BritishColumbia(cities=value['British Columbia'], directory=directory)
    country = Canada(provinces=[bc])

    for city in bc.cities:
        city.update_databases(net=False)
        city.centrality(run=False, axial=True, layer='network_walk')
        city.centrality(run=False, osm=True, layer='network_drive')
        city.node_elevation(run=False)
        gdf = aerial_buffer(gpd.read_file(city.gpkg, layer='land_dissemination_area'),
                            '/Volumes/Samsung_T5/Databases/Metro Vancouver, British Columbia')
        gdf.to_feather(f'{directory}Network/{city.municipality}_ab.feather')
        network_analysis = city.network_analysis(
            prefix='mob',
            run=True,
            service_areas=radii,
            sample_gdf=gpd.read_file(city.gpkg, layer='land_dissemination_area'),
            decays=['flat'],
            filter_min={'population density per square kilometre, 2016': 300},
            aggregated_layers=network_layers,
            keep=['dauid', 'walk', 'bike', 'drive', 'bus', 'geometry'])
        gc.collect()
