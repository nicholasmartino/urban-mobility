from Geospatial import City, Sandbox


# Process mobility related indicators for elementslab
regions = {
    # 'yvr': ['Metro Vancouver, British Columbia', 'Sunset'],
    'yxs': ['Prince George, British Columbia', 'West Bowl'],
    # 'yyj': ['Capital Regional District, British Columbia', 'Hillside Quadra']
}

for key, value in regions.items():
    city = City(municipality=value[0])
    city.check_file_databases(bound=False, net=False, census=True, bcaa=False, icbc=False)
    city.set_parameters(unit='lda', service_areas=[400, 800, 1600], samples=None)
    # city.density_indicators()
    # city.diversity_indicators()
    # city.street_network_indicators()
    # city.cycling_network_indicators()
    sandbox = Sandbox(name=value[1])
    sandbox.morph_indicators()
