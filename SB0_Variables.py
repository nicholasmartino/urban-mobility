windows = False
if windows:
    directory = '//nas.sala.ubc.ca/ELabs/100_personal/nm/Databases/'
    bca_dir = '//nas.sala.ubc.ca/ELabs/50_projects/16_PICS/07_BCA data/'
else:
    directory = '/Volumes/Samsung_T5/Databases/'
    bca_dir = '/Volumes/ELabs/50_projects/16_PICS/07_BCA data/'

modes = ['transit', 'bike', 'walk', 'drive']
radii = [1600, 1200, 800, 400]
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
    # 'West Bowl': ['Prince George, British Columbia', wb_experiments],
    # 'Hillside Quadra': ['Capital Regional District, British Columbia', hq_experiments],
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
    'network_walk': ["walk_length", "walk_straight"],
    'network_bike': ["bike_length", "bike_straight"],
    'network_drive': ["drive_length", "drive_straight"],
    'land_assessment_parcels': ["area_sqkm", "n_size"],
    'land_dissemination_area': ["population, 2016", "population density per square kilometre, 2016", "n_dwellings"],
    'land_assessment_fabric': ["n_use", "number_of_bedrooms"],
}

rename_dict = {
    'mob_network_stops_ct': ('Public transit stops', 'network'),
    'frequency': ('Transit frequency', 'network'),
    'network_nodes': ('Number of intersections', 'network'),
    'elevation': ('Elevation', 'network'),

    'connectivity': ('Axial connectivity', 'network'),
    'axial_closeness': ('Axial closeness centrality', 'network'),
    'axial_betweenness': ('Axial betweenness centrality', 'network'),
    'axial_n_betweenness': ('Normalized axial betweenness centrality', 'network'),
    'axial_length': ('Axial line length', 'network'),
    'axial_eigenvector': ('Axial eigenvector centrality', 'network'),
    'axial_katz': ('Axial katz centrality', 'network'),
    'axial_pagerank': ('Axial page rank centrality', 'network'),
    'axial_hits1': ('Axial hits centrality', 'network'),
    'axial_degree': ('Axial degree centrality', 'network'),

    'network_walk_ct': ('Intensity of walkable Network', 'network'),
    'network_bike_ct': ('Intensity of bikeable Network', 'network'),
    'network_drive_ct': ('Intensity of driveable Network', 'network'),
    'walk_length': ('Length of walkable Network', 'network'),
    'bike_length': ('Length of bikeable Network', 'network'),
    'drive_length': ('Length of driveable Network', 'network'),
    'walk_straight': ('Straightness of walkable Network', 'network'),
    'bike_straight': ('Straightness of bikeable Network', 'network'),
    'drive_straight': ('Straightness of driveable Network', 'network'),

    'land_assessment_fabric_ct': ('Number of units', 'density'),
    'n_use': ('Use diversity', 'landuse'),
    'CM': ('Commercial', 'landuse'),
    'SFD': ('Single-Family Detached', 'landuse'),
    'SFA': ('Single-Family Attached', 'landuse'),
    'MFL': ('Multi-Family Low-Rise', 'landuse'),
    'MFH': ('Multi-Family High-Rise', 'landuse'),
    'total_finished_area': ('Total finished area', 'density'),
    'gross_building_area': ('Gross building area', 'density'),
    'number_of_bedrooms': ('Number of bedrooms', 'density'),

    'land_assessment_parcels_ct': ('Number of parcels', 'density'),
    'area_sqkm': ('Parcel size', 'density'),
    'n_size': ('Parcel diversity', 'density'),

    'population density per square kilometre, 2016': ('Population density', 'density'),
    'n_dwellings': ('Number of dwellings', 'density'),
    'population, 2016': ('Population', 'density'),

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
