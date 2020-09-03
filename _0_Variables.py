regions = {
    'Canada':
        {
            'British Columbia': [
                'Capital Regional District',
                'Metro Vancouver',
            ]
        }
}

network_layers = {
    'network_stops': ["frequency"],
    'network_links': ["length"],
    'network_nodes': ["elevation", "node_closeness", "node_betweenness", "node_n_betweenness"],
    'network_axial': ["axial_degree", "axial_closeness", "axial_betweenness", "axial_eigenvector",
        "axial_katz", "axial_length", "axial_pagerank", "axial_hits1"],
    'network_drive': ["length", "link_betweenness", "link_n_betweenness"],
    'network_cycle': ["cycle_length"],
    'land_assessment_fabric': ["n_use", "year_built", "total_finished_area", "gross_building_area",
        "number_of_bedrooms", "number_of_bathrooms"],
    'land_assessment_parcels': ["area_sqkm", "n_size"],
    'land_dissemination_area': ["population, 2016", "population density per square kilometre, 2016",
        "n_dwellings", "dwelling_div_rooms_si", "dwelling_div_bedrooms_si", "dwelling_div_rooms_sh",
        "dwelling_div_bedrooms_sh"]
}

radius = [1200, 1000, 800, 600, 400] #4800, 3200, 1600]

sandboxes = ['Sunset']