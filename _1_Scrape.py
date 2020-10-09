from Geospatial.Scraper import BritishColumbia, Canada
from _0_Variables import regions

# Download and transfer data
for key, value in regions.items():
    bc = BritishColumbia(cities=value['British Columbia'])
    country = Canada(provinces=[bc])

    # StatsCan
    country.update_databases(census=False)

    # OpenStreetMaps
    for city in bc.cities: city.update_databases(bound=False, net=False)

    # BC Assessment
    windows = False
    if windows: bca_dir = '//nas.sala.ubc.ca/ELabs/50_projects/16_PICS/07_BCA data/'
    else: bca_dir = '/Volumes/ELabs/50_projects/16_PICS/07_BCA data/'
    bc.aggregate_bca_from_field(
        run=False, join=True, classify=True,
        inventory_dir=f'{bca_dir}170811_BCA_Provincial_Data/Inventory Information - RY 2017.csv',
        geodatabase_dir=f'{bca_dir}Juchan_backup/BCA_2017_roll_number_method/BCA_2017_roll_number_method.gdb')

    # BC Transit
    bc.get_bc_transit(run=True, down=False, urls=[
        #'http://victoria.mapstrat.com/current/google_transit.zip',
        'local', 'local'
        #'https://www.bctransit.com/data/gtfs/prince-george.zip'
    ])
