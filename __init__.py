from Geospatial import City


# Process mobility related indicators for elementslab
regions = {
    # 'yvr': 'Metro Vancouver, British Columbia',
    'yxs': 'Prince George, British Columbia',
    # 'yyj': 'Capital Regional District, British Columbia'
}

for key, value in regions.items():
    city = City(municipality=value)
    city.check_file_databases(bound=False, net=False, census=True, bcaa=False, icbc=False)
    city.set_parameters(unit='lda', service_areas=[400, 800, 1600], samples=None)
    # city.density_indicators()
    # city.diversity_indicators()
    # city.street_network_indicators()
    # city.cycling_network_indicators()
    # city.geomorph_indicators()
    # city.gravity()
    city.linear_correlation_lda()
