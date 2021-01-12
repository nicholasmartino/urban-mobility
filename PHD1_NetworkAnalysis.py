import gc
import geopandas as gpd
from UrbanScraper.Local import BritishColumbia, Network


radii = [1600, 800]
network_layers = {
    'network_stops': ["frequency"],
    'network_links': ["length"],
    'network_nodes': ["elevation"],
    'network_axial': ["axial_degree", "axial_closeness", "axial_betweenness", "axial_eigenvector",
        "axial_katz", "axial_length", "axial_pagerank", "axial_hits1"],
    'network_drive': ["length"],
    'network_cycle': ["cycle_length"],
    'land_osm_amenities_f': ["amenity"],
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

# Filter amenities to avoid sigkill
amenities2keep = [
	'bench',
	'restaurant',
	'bicycle_parking',
	'cafe',
	'fast_food',
	'waste_basket',
	'post_box',
	'toilets',
	'bank',
	'drinking_water',
	'parking',
	'pharmacy',
	'bicycle_rental',
	'dentist',
	'fuel',
	'pub',
	'post_office',
	'bar',
	'public_bookcase',
	'clinic',
	'place_of_worship',
	'car_sharing',
	'atm',
	'recycling',
	'waste_disposal',
	'school',
	'charging_station',
	'community_centre',
	'shelter',
	'library',
	'theatre',
	'doctors',
	'ferry_terminal',
	'car_rental',
	'social_facility',
	'childcare',
	'police',
	'car_wash',
	'bus_station',
	'fire_station',
	'food_court',
	'marketplace',
	'college',
	'arts_centre',
	'cinema',
	'hospital',
	'university',
]

amenities = gpd.read_file(city.gpkg, layer='land_osm_amenities')
amenities = amenities[amenities['amenity'].isin(amenities2keep)].loc[:, ['amenity', 'geometry']]
amenities.to_file(city.gpkg, layer='land_osm_amenities_f')

# Aggregation for unsupervised clustering that'll be used for site selection
city.network_analysis(
	prefix='aff_lda',
	run=False,
	service_areas=[1600],
	sample_gdf=gpd.read_file(city.gpkg, layer='land_dissemination_area'),
	decays=['flat', 'linear'],
	aggregated_layers={
		'craigslist_rent': ["price_sqft"],
		'land_osm_amenities_f': ["amenity"],
		'land_dissemination_area': ['walk', 'bike', 'bus', 'drive']
	},
)

# Aggregation for supervised regression that'll be used for rent price and mode share prediction
for radius in radii:
	city.network_analysis(
    prefix=f'aff_{radius}',
    run=True,
    service_areas=[radius],
    sample_gdf=gpd.read_feather(f'/Volumes/Samsung_T5/Databases/Network/Metro Vancouver, British Columbia_parcels.feather'),
    decays=['flat'],
    aggregated_layers=network_layers,
	operations=['count', 'sum', 'mean'],
	diversity=False
)
gc.collect()
