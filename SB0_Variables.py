windows = False
if windows:
    directory = '//nas.sala.ubc.ca/ELabs/100_personal/nm/Databases/'
    bca_dir = '//nas.sala.ubc.ca/ELabs/50_projects/16_PICS/07_BCA data/'
else:
    directory = '/Volumes/Samsung_T5/Databases/'
    bca_dir = '/Volumes/ELabs/50_projects/16_PICS/07_BCA data/'

modes = ['transit', 'bike', 'walk', 'drive']
radii = [1200, 800, 400]
r_seeds = 6
trip_length = {'transit': 10.2, 'drive': 10.4, 'bike': 5.5, 'walk': 0.9}

ss_experiments = {
    'e0': 2020,
    # 'e1': 2030,
    # 'e2': 2030,
    'e3': 2040,
    # 'e4': 2040,
    'e5': 2040,
    'e6': 2040,
    # 'e7': 2050,
    # 'e8': 2050
}
wb_experiments = {
    'e0': 2020,
    'e5': 2050,
    'e6': 2050,
    'e7': 2050,
}
hq_experiments = {
    'e0': 2020,
    'e1': 2040,
    'e2': 2040,
    'e3': 2040
}
experiments = {
    'West Bowl': ['Prince George, British Columbia', wb_experiments],
    'Hillside Quadra': ['Capital Regional District, British Columbia', hq_experiments],
    'Sunset': ['Metro Vancouver, British Columbia', ss_experiments],
}

regions = {
    'Canada': {
        'British Columbia': [v[0].split(',')[0] for i, v in experiments.items()]
    }
}

network_layers = {
    'network_stops': ["frequency"],
    'network_nodes': ["elevation"],
    # 'network_axial': ["connectivity", "axial_closeness", "axial_betweenness", "axial_n_betweenness", "axial_length",
    #     "axial_eigenvector", "axial_katz", "axial_pagerank", "axial_hits1", "axial_degree"],
    'network_walk': ["walk_length", "walk_straight"],
    'network_bike': ["bike_length", "bike_straight"],
    'network_drive': ["drive_length", "drive_straight"],
    'land_assessment_fabric': ["n_use", "total_finished_area", "gross_building_area", "number_of_bedrooms"],
    'land_assessment_parcels': ["area_sqkm", "n_size"],
    'land_dissemination_area': ["population, 2016", "population density per square kilometre, 2016", "n_dwellings"]
}

"""
# MACC Curve

network_bus = {
    'network_stops': ["frequency"],
    'land_assessment_fabric': ["n_use", "total_finished_area"],  # "gross_building_area", "year_built", "number_of_bedrooms"],
    'land_assessment_parcels': ["area_sqkm", "n_size"],
    'land_dissemination_area': ["population, 2016", "population density per square kilometre, 2016"], # "n_dwellings"]
}

network_bike = {
    'network_cycle': ["cycle_length"],
    'land_assessment_fabric': ["n_use", "total_finished_area"],  # "gross_building_area", "year_built", "number_of_bedrooms"],
    'land_assessment_parcels': ["area_sqkm", "n_size"],
    'land_dissemination_area': ["population, 2016", "population density per square kilometre, 2016"], # "n_dwellings"]
}
"""
