from Geospatial import City, Sandbox


# Process mobility related indicators for elementslab
regions = {
    # 'yvr': ['Metro Vancouver, British Columbia', 'Sunset'],
    'yxs': ['Prince George, British Columbia', 'West Bowl', 'west_bowl.gdb', [
        'E0_2020_PRCLS',
        'E5_2050_NC_FOCUS_PRCLS',
        'E6_2050_COR_FOCUS_PRCLS'
    ]],
    # 'yyj': ['Capital Regional District, British Columbia', 'Hillside Quadra']
}

for key, value in regions.items():
    city = City(municipality=value[0])
    city.update_databases(bound=True, net=False, census=False, bcaa=False, icbc=False)
    # city.aggregate_bca_from_field()
    city.set_parameters(unit='lda', service_areas=[400, 800, 1600], samples=None)
    city.regression()
    # city.export_map()
    city.network_analysis(sample='lda', service_areas=[400, 800, 1600], cols={})
    # city.density_indicators()
    # city.diversity_indicators()
    # city.street_network_indicators()
    # city.cycling_network_indicators()
    # city.export_databases()
    # city.export_destinations()
    sandbox = Sandbox(name=value[1], geodatabase=value[2], layers=value[3])
    sandbox.morph_indicators()
