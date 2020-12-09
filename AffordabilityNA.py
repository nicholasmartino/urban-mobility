import gc
import geopandas as gpd
from UrbanScraper.Local import BritishColumbia


radius = [1600, 1200, 800, 400] #4800, 3200, 1600]
network_layers = {
    'network_stops': ["frequency"],
    'network_links': ["length"],
    'network_nodes': ["elevation"], # "node_betweenness", "node_n_betweenness", "node_closeness"],
    'network_axial': ["axial_degree", "axial_closeness", "axial_betweenness", "axial_eigenvector",
        "axial_katz", "axial_length", "axial_pagerank", "axial_hits1"],
    'network_drive': ["length"], #, "link_betweenness", "link_n_betweenness"],
    'network_cycle': ["cycle_length"],
    'land_osm_amenities': ["amenity"],
    'land_assessment_fabric': ["n_use", "year_built", "total_finished_area", "gross_building_area",
        "number_of_bedrooms"],
    'land_assessment_parcels': ["area_sqkm", "n_size"],
    'land_dissemination_area': ["population, 2016", "population density per square kilometre, 2016",
        "n_dwellings"]
}

# Perform network analysis for the real place
bc = BritishColumbia(cities=['Metro Vancouver'])

for city in bc.cities:
	city.centrality(run=False, axial=True, layer='network_walk')
	city.centrality(run=False, osm=True, layer='network_drive')
	city.node_elevation(run=False)
	network_analysis = city.network_analysis(
	    prefix='aff',
	    run=True,
	    service_areas=radius,
	    sample_gdf=gpd.read_feather(f'/Volumes/Samsung_T5/Databases/Network/Metro Vancouver, British Columbia_parcels.feather'),
	    decays=['flat', 'linear'],
	    aggregated_layers=network_layers,
	)
	gc.collect()
