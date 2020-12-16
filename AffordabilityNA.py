import gc
import geopandas as gpd
from UrbanScraper.Local import BritishColumbia, Network


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
city = Network(f"Metro Vancouver, British Columbia")
city.centrality(run=False, axial=True, layer='network_walk')
city.centrality(run=False, osm=True, layer='network_drive')
city.node_elevation(run=False)

city.network_analysis(
	prefix='aff_lda',
	run=True,
	service_areas=[1600],
	sample_gdf=gpd.read_file(city.gpkg, layer='land_dissemination_area'),
	decays=['flat', 'linear'],
	aggregated_layers={
		'craigslist_rent': ["price_sqft"],
		'land_osm_amenities': ["amenity"],
		'land_dissemination_area': ['walk', 'bike', 'bus', 'drive']
	},
)

city.network_analysis(
    prefix='aff',
    run=False,
    service_areas=radius,
    sample_gdf=gpd.read_feather(f'/Volumes/Samsung_T5/Databases/Network/Metro Vancouver, British Columbia_parcels.feather'),
    decays=['flat', 'linear'],
    aggregated_layers=network_layers,
)
gc.collect()
