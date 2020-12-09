from UrbanScraper.Scraper import Canada
from UrbanScraper.Local import BritishColumbia
from _0_Variables import *


# Download and transfer data
for key, value in regions.items():
    bc = BritishColumbia(cities=value['British Columbia'], directory=directory)
    country = Canada(provinces=[bc])

    # StatsCan
    country.update_databases(census=True)

    # OpenStreetMaps
    for city in bc.cities: city.update_databases(bound=True, net=True)

    # BC Assessment
    bc.aggregate_bca_from_field(
        run=True, join=True, classify=True,
        inventory_dir=f'{bca_dir}170811_BCA_Provincial_Data/Inventory Information - RY 2017.csv',
        geodatabase_dir=f'{bca_dir}Juchan_backup/BCA_2017_roll_number_method/BCA_2017_roll_number_method.gdb')

    # BC Transit
    bc.get_bc_transit(run=True, down=True, urls=[
        'http://victoria.mapstrat.com/current/google_transit.zip',
        'local', 'local'
        'https://www.bctransit.com/data/gtfs/prince-george.zip'
    ])
