"""
MIT License

Copyright (c) 2020 Nicholas Martino

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import datetime
import math
import glob, os
os.environ["PROJ_LIB"] = r"C:\\Users\\Anaconda3\\Library\\share"
import time
import timeit
import warnings
from shutil import copyfile
import pandana as pdna
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from Statistics.basic_stats import shannon_div
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.colors import ListedColormap
from matplotlib import cm
# from mpl_toolkits.basemap import Basemap


class City:
    def __init__(self, directory='../Geospatial/Databases/',
                 municipality='City, State',
                 census_file='Statistics Canada.gpkg',
                 tax_assessment_file='BC Assessment.gpkg'):
        self.municipality = municipality
        self.directory = directory
        self.census_file = self.directory + census_file
        self.assessment = self.directory + tax_assessment_file
        self.gpkg = self.directory + self.municipality + '.gpkg'
        self.city_name = self.municipality.partition(',')[0]
        self.crs = '26910'
        try:
            self.edges = gpd.read_file(self.gpkg, layer='network_streets')
            self.nodes = gpd.read_file(self.gpkg, layer='network_intersections')
        except: pass
        try:
            self.boundary = gpd.read_file(self.gpkg, layer='land_municipal_boundary')
            self.bbox = self.boundary.total_bounds
        except: print('land_municipal_boundary layer not read')
        try: self.LDAs = gpd.read_file(self.gpkg, layer='land_dissemination_area')
        except: print('land_dissemination_area layer not read')
        try: self.parcels = gpd.read_file(self.gpkg, layer='land_assessment_fabric')
        except: print('land_assessment_fabric layer not read')
        try:
            self.walking_net = gpd.read_file(self.gpkg, layer='network_streets_walking')
            self.cycling_net = gpd.read_file(self.gpkg, layer='network_streets_cycling')
            self.driving_net = gpd.read_file(self.gpkg, layer='network_streets_driving')
        except: pass
        try: self.cycling = gpd.read_file(self.gpkg, layer='network_cycling')
        except: print('network_cycling layer not read')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=DeprecationWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        warnings.simplefilter(action='ignore', category=ResourceWarning)
        print('Class ' + self.city_name + ' created @ ' + str(datetime.datetime.now()))

    # Scrape and clean data
    def update_databases(self, bound=True, net=True, census=True, bcaa=True, icbc=True):
        # Check if boundary data exists and download it from OSM if not
        if bound:
            try:
                self.boundary = gpd.read_file(self.gpkg, layer='land_municipal_boundary')
                self.bbox = self.boundary.total_bounds
                print(self.city_name + ' boundary read from database')
            except:
                print('No boundary in database, downloading for ' + self.city_name)
                self.boundary = ox.gdf_from_place(self.municipality)  # gdf
                self.boundary.crs = {'init': 'epsg:4326'}
                self.boundary.to_crs({'init': 'epsg:26910'}, inplace=True)
                self.boundary.to_file(self.gpkg, layer='land_municipal_boundary', driver='GPKG')
                self.boundary = gpd.read_file(self.gpkg, layer='land_municipal_boundary')
                self.bbox = self.boundary.total_bounds
            s_index = self.boundary.sindex

        # Check if network data exists and download it from OSM if not
        if net:
            try:
                self.edges = gpd.read_file(self.gpkg, layer='network_streets')
                self.nodes = gpd.read_file(self.gpkg, layer='network_intersections')
                print(self.city_name + ' network read from database')
            except:
                print('No network in database, downloading for ' + self.city_name)
                network = ox.graph_from_place(self.municipality)
                ox.save_graph_shapefile(network, 'osm', self.directory)
                edges = gpd.read_file(self.directory+'osm/edges/edges.shp')
                nodes = gpd.read_file(self.directory+'osm/nodes/nodes.shp')
                edges.crs = {'init': 'epsg:4326'}
                edges.to_crs(crs='epsg:26910', inplace=True)
                edges.to_file(self.gpkg, layer='network_streets', driver='GPKG')
                nodes.crs = {'init': 'epsg:4326'}
                nodes.to_crs(crs='epsg:26910', inplace=True)
                nodes.to_file(self.gpkg, layer='network_intersections', driver='GPKG')
                try:
                    os.remove('osm')
                except:
                    pass
                self.edges = gpd.read_file(self.gpkg, layer='network_streets')
                self.nodes = gpd.read_file(self.gpkg, layer='network_intersections')

        # Check if dissemination data exists and join it from census database if not
        if census:
            try:
                self.LDAs = gpd.read_file(self.gpkg, layer='land_dissemination_area')
                print(self.city_name + ' census data read from database')
            except:
                census_lda = gpd.read_file(self.census_file, layer='land_dissemination_area')
                census_lda.crs = {'init': 'epsg:4326'}
                census_lda.to_crs({'init': 'epsg:26910'}, inplace=True)
                self.LDAs = gpd.sjoin(census_lda, self.boundary)

                # Calculate population density
                fields = ['Car; truck; van - as a driver', 'Car; truck; van - as a passenger',
                          'Public transit', 'Walked', 'Bicycle', 'Other method']
                fgdf = gpd.GeoDataFrame()
                for field in fields:
                    fgdf[field] = self.LDAs[field].astype('float')
                self.LDAs['pop'] = fgdf.sum(axis=1, skipna=True)
                self.LDAs['pop_den'] = self.LDAs['pop'] / (self.LDAs.geometry.area * 10000)

                self.LDAs.to_file(self.gpkg, layer='land_dissemination_area')
                print('Census dissemination area joined for ' + self.city_name)

        # Check if BC Assessment data exists and join it from BCAA database if not
        if bcaa:
            try:
                self.parcels = gpd.read_file(self.gpkg, layer='land_assessment_fabric')
                print(self.city_name + ' BC Assessment data read from database')
            except:
                # # Spatial join with spatial index BC Assessment data
                start_time = time.time()
                # gdf = gpd.read_file(self.assessment, layer='land_assessment_fabric')
                # print("BC Assessment data read in %s minutes" % str(
                #     round((time.time() - start_time) / 60, 2)))
                # start_time = time.time()
                # matches = gpd.sjoin(gdf, self.boundary, op='within')
                # matches.to_file(self.gpkg, layer='land_assessment_fabric')
                self.parcels = self.aggregate_bca_from_field()
                print("Spatial join and export for " + self.city_name + " performed in %s minutes " % str(
                    round((time.time() - start_time) / 60, 2)))

        # Check if ICBC crash data exists and join it from ICBC database if not
        if icbc:
            try:
                self.crashes = gpd.read_file(self.gpkg, layer='network_accidents')
                print(self.city_name + ' ICBC data read from database')
            except:
                source = 'https://public.tableau.com/profile/icbc#!/vizhome/LowerMainlandCrashes/LMDashboard'
                print('Adding ICBC crash data to ' + self.city_name + ' database')
                df = self.merge_csv('Databases/ICBC/')
                geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
                gdf = gpd.GeoDataFrame(df, geometry=geometry)
                gdf.crs = {'init': 'epsg:4326'}
                gdf.to_crs({'init': 'epsg:26910'}, inplace=True)
                matches = gpd.sjoin(gdf, self.boundary, op='within')
                matches.to_file(self.gpkg, layer='network_accidents', driver='GPKG')

    def aggregate_bca_from_field(self):
        print('Aggregating BC Assessment from field')
        inventory = self.directory+'BCA/Inventory Information - RY 2017.csv'
        df = pd.read_csv(inventory)

        # Load and process Roll Number field on both datasets
        gdf = gpd.read_file(self.directory+'BCA/BCA_2017_roll_number_method.gdb', layer='ASSESSMENT_FABRIC')
        gdf.crs = {'init': 'epsg:3005'}
        gdf.to_crs({'init': 'epsg:26910'}, inplace=True)
        s_index = gdf.sindex
        gdf = gpd.sjoin(gdf, self.boundary, op='within')
        gdf['JUROL'] = gdf['JUROL'].astype(str)
        gdf = gdf[gdf.geometry.area > 71]
        print('BCA spatial layer loaded with ' + str(len(gdf)) + ' parcels')
        df['JUR'] = df['JUR'].astype(int).astype(str)
        df['ROLL_NUM'] = df['ROLL_NUM'].astype(str)
        df['JUROL'] = df['JUR'] + df['ROLL_NUM']

        merged = pd.merge(gdf, df, on='JUROL')
        full_gdfs = {'0z': merged}
        print(': ' + str(len(full_gdfs['0z'])))

        for i in range(1, 7):
            strings = []
            for n in range(i):
                strings.append('0')
            string = str(''.join(strings))
            df[string + 'z'] = string
            df['JUROL'] = df['JUR'] + string + df['ROLL_NUM']
            full_gdf = pd.merge(gdf, df, on='JUROL')
            full_gdf.drop([string + 'z'], axis=1)
            if len(full_gdf) > 0:
                full_gdfs[str(i) + 'z'] = full_gdf
            print(string + ': ' + str(len(full_gdf)))

        # Merge and export spatial and non-spatial datasets
        out_gdf = pd.concat(full_gdfs.values(), ignore_index=True)
        print(len(out_gdf))

        # Reclassify land uses for BC Assessment data
        uses = {
            'residential': ['000 - Single Family Dwelling', '030 - Strata-Lot Residence (Condominium)',
                            '032 - Residential Dwelling with Suite',
                            '033 - Duplex, Non-Strata Side by Side or Front / Back',
                            '034 - Duplex, Non-Strata Up / Down', '035 - Duplex, Strata Side by Side',
                            '036 - Duplex, Strata Front / Back', '039 - Row Housing (Single Unit Ownership)',
                            '037 - Manufactured Home (Within Manufactured Home Park)',
                            '038 - Manufactured Home (Not In Manufactured Home Park)',
                            '040 - Seasonal Dwelling',
                            '041 - Duplex, Strata Up / Down', '047 - Triplex', '049 - Fourplex',
                            '050 - Multi-Family (Apartment Block)',
                            '052 - Multi-Family (Garden Apartment & Row Housing)', '053 - Multi-Family (Conversion)',
                            '054 - Multi-Family (High-Rise)', '055 - Multi-Family (Minimal Commercial)',
                            '056 - Multi-Family (Residential Hotel)', '057 - Stratified Rental Townhouse',
                            '058 - Stratified Rental Apartment (Frame Construction)',
                            '059 - Stratified Rental Apartment (Hi-Rise Construction)',
                            '060 - 2 Acres Or More (Single Family Dwelling, Duplex)', '285 - Seniors Licensed Care',
                            '062 - 2 Acres Or More (Seasonal Dwelling)',
                            '063 - 2 Acres Or More (Manufactured Home)',
                            '234 - Manufactured Home Park',
                            '286 - Seniors Independent & Assisted Living'],
            'vacant': ['001 - Vacant Residential Less Than 2 Acres', '051 - Multi-Family (Vacant)',
                       '061 - 2 Acres Or More (Vacant)', '201 - Vacant IC&I',
                       '421 - Vacant', '422 - IC&I Water Lot (Vacant)',
                       '601 - Civic, Institutional & Recreational (Vacant)'],
            'parking': ['020 - Residential Outbuilding Only', '029 - Strata Lot (Parking Residential)',
                        '043 - Parking (Lot Only, Paved Or Gravel-Res)', '219 - Strata Lot (Parking Commercial)',
                        '260 - Parking (Lot Only, Paved Or Gravel-Com)', '262 - Parking Garage',
                        '490 - Parking Lot Only (Paved Or Gravel)'],
            'other': ['002 - Property Subject To Section 19(8)', '070 - 2 Acres Or More (Outbuilding)', '190 - Other',
                      '200 - Store(S) And Service Commercial', '205 - Big Box', '216 - Commercial Strata-Lot',
                      '220 - Automobile Dealership', '222 - Service Station', '224 - Self-Serve Service Station',
                      '226 - Car Wash', '227 - Automobile Sales (Lot)', '228 - Automobile Paint Shop, Garages, Etc.',
                      '230 - Hotel', '232 - Motel & Auto Court', '233 - Individual Strata Lot (Hotel/Motel)',
                      '237 - Bed & Breakfast Operation 4 Or More Units',
                      '239 - Bed & Breakfast Operation Less Than 4 Units',
                      '240 - Greenhouses And Nurseries (Not Farm Class)', '257 - Fast Food Restaurants',
                      '258 - Drive-In Restaurant', '288 - Sign Or Billboard Only'],
            'retail': ['202 - Store(S) And Living Quarters', '209 - Shopping Centre (Neighbourhood)',
                       '211 - Shopping Centre (Community)', '212 - Department Store - Stand Alone',
                       '213 - Shopping Centre (Regional)', '214 - Retail Strip', '215 - Food Market',
                       '225 - Convenience Store/Service Station'],
            'entertainment': ['236 - Campground (Commercial)', '250 - Theatre Buildings',
                              '254 - Neighbourhood Pub', '256 - Restaurant Only',
                              '266 - Bowling Alley', '270 - Hall (Community, Lodge, Club, Etc.)',
                              '280 - Marine Facilities (Marina)',
                              '600 - Recreational & Cultural Buildings (Includes Curling',
                              '610 - Parks & Playing Fields', '612 - Golf Courses (Includes Public & Private)',
                              '654 - Recreational Clubs, Ski Hills',
                              '660 - Land Classified Recreational Used For'],
            'civic': ['210 - Bank', '620 - Government Buildings (Includes Courthouse, Post Office',
                              '625 - Garbage Dumps, Sanitary Fills, Sewer Lagoons, Etc.', '630 - Works Yards',
                              '634 - Government Research Centres (Includes Nurseries &',
                              '640 - Hospitals (Nursing Homes Refer To Commercial Section).',
                              '642 - Cemeteries (Includes Public Or Private).',
                              '650 - Schools & Universities, College Or Technical Schools',
                              '652 - Churches & Bible Schools'],
            'agriculture': ['110 - Grain & Forage', '120 - Vegetable & Truck',
                            '150 - Beef', '170 - Poultry', '180 - Mixed'],
            'office': ['203 - Stores And/Or Offices With Apartments', '204 - Store(S) And Offices',
                       '208 - Office Building (Primary Use)'],
            'industrial': ['217 - Air Space Title', '272 - Storage & Warehousing (Open)',
                           '273 - Storage & Warehousing (Closed)', '274 - Storage & Warehousing (Cold)',
                           '275 - Self Storage', '276 - Lumber Yard Or Building Supplies', '400 - Fruit & Vegetable',
                           '401 - Industrial (Vacant)', '402 - Meat & Poultry', '403 - Sea Food',
                           '404 - Dairy Products', '405 - Bakery & Biscuit Manufacturing',
                           '406 - Confectionery Manufacturing & Sugar Processing', '408 - Brewery',
                           '414 - Miscellaneous (Food Processing)',
                           '416 - Planer Mills (When Separate From Sawmill)',
                           '419 - Sash & Door',
                           '420 - Lumber Remanufacturing (When Separate From Sawmill)',
                           '423 - IC&I Water Lot (Improved)',
                           '424 - Pulp & Paper Mills (Incl Fine Paper, Tissue & Asphalt Roof)',
                           '425 - Paper Box, Paper Bag, And Other Paper Remanufacturing.', '428 - Improved',
                           '429 - Miscellaneous (Forest And Allied Industry)',
                           '434 - Petroleum Bulk Plants',
                           '445 - Sand & Gravel (Vacant and Improved)',
                           '447 - Asphalt Plants',
                           '448 - Concrete Mixing Plants',
                           '449 - Miscellaneous (Mining And Allied Industries)', '452 - Leather Industry',
                           '454 - Textiles & Knitting Mills', '456 - Clothing Industry',
                           '458 - Furniture & Fixtures Industry', '460 - Printing & Publishing Industry',
                           '462 - Primary Metal Industries (Iron & Steel Mills,', '464 - Metal Fabricating Industries',
                           '466 - Machinery Manufacturing (Excluding Electrical)',
                           '470 - Electrical & Electronics Products Industry',
                           '472 - Chemical & Chemical Products Industries', '474 - Miscellaneous & (Industrial Other)',
                           '476 - Grain Elevators', '478 - Docks & Wharves', '500 - Railway',
                           '505 - Marine & Navigational Facilities (Includes Ferry',
                           '510 - Bus Company, Including Street Railway', '520 - Telephone',
                           '530 - Telecommunications (Other Than Telephone)',
                           '550 - Gas Distribution Systems',
                           '560 - Water Distribution Systems',
                           '580 - Electrical Power Systems (Including Non-Utility']
        }

        new_uses = []
        index = list(out_gdf.columns).index("PRIMARY_ACTUAL_USE")
        all_prim_uses = [item for sublist in list(uses.values()) for item in sublist]
        for row in out_gdf.iterrows():
            for key, value in uses.items():
                if row[1]['PRIMARY_ACTUAL_USE'] in value:
                    new_uses.append(key)
            if row[1]['PRIMARY_ACTUAL_USE'] not in all_prim_uses:
                new_uses.append(row[1]['PRIMARY_ACTUAL_USE'])
        out_gdf['elab_use'] = new_uses

        out_gdf.to_file(self.gpkg, driver='GPKG', layer='land_assessment_fabric')
        return out_gdf

    def aggregate_bca_from_location(self):
        # WIP
        gdf = gpd.read_file(self.assessment, layer='land_assessment_fabric')
        centroids = []
        for geom in gdf.geometry:
            print(geom)
            centroids.append(geom.centroid)
        coords = []
        for pt in centroids:
            print(pt)
            coords.append(str(pt.x) + '_' + str(pt.y))
        gdf['CID'] = coords
        df = gdf.groupby('CID').agg({
            'JUROL': 'first',
            'YEAR_BUILT': 'max',
            'NUMBER_OF_STOREYS': 'max',
            'PRIMARY_ACTUAL_USE': 'first',
            'NUMBER_OF_BATHROOMS': 'sum',
            'NUMBER_OF_BEDROOMS': 'sum',
            'STRATA_UNIT_AREA': 'sum',
            'TOTAL_FINISHED_AREA': 'sum',
            'GROSS_BUILDING_AREA': 'mean',
            'LAND_SIZE': 'mean',
            'LAND_DEPTH': 'mean',
            'ACTUAL_LAND': 'mean',
            'ACTUAL_IMPR': 'mean',
            'ACTUAL_TOTAL': 'mean'})
        df['CID'] = df.index
        gdf_geo = gdf.drop_duplicates("CID", keep='first')
        print(str(len(df)) + ' aggregated rows for ' + str(len(gdf_geo.geometry)) + ' geometries')
        out_geoms = []
        for geo in gdf_geo.geometry:
            out_geoms.append(geo)
        df['geometry'] = out_geoms
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs={'init': 'epsg:3005'})
        gdf.to_file(self.assessment, layer='land_assessment_fabric_agg', driver='GPKG')

    def filter_networks(self):
        # Filter Open Street Map Networks into Walking, Cycling and Driving
        walking = ['bridleway', 'corridor', 'footway', 'living_street', 'path', 'pedestrian', 'residential',
                   'road', 'secondary', 'service', 'steps', 'tertiary', 'track', 'trunk']
        self.walking_net = self.edges.loc[self.edges.highway.apply(lambda x: any(element in x for element in walking))]

        cycling = ['bridleway', 'corridor', 'cycleway', 'footway', 'living_street', 'path', 'pedestrian',
                   'residential', 'road', 'secondary', 'service', 'tertiary', 'track', 'trunk']
        self.cycling_net = self.edges.loc[self.edges.highway.apply(lambda x: any(element in x for element in cycling))]

        driving = ['corridor', 'living_street', 'motorway', 'primary', 'primary_link', 'residential', 'road',
                   'secondary', 'secondary_link', 'service', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link',
                   'unclassified']
        self.driving_net = self.edges.loc[self.edges.highway.apply(lambda x: any(element in x for element in driving))]

        self.walking_net.to_file(self.gpkg, layer='network_streets_walking')
        self.cycling_net.to_file(self.gpkg, layer='network_streets_cycling')
        self.driving_net.to_file(self.gpkg, layer='network_streets_driving')
        return None

    def elevation(self, hgt_file, lon, lat):
        SAMPLES = 1201  # Change this to 3601 for SRTM1
        with open(hgt_file, 'rb') as hgt_data:
            # Each data is 16bit signed integer(i2) - big endian(>)
            elevations = np.fromfile(hgt_data, np.dtype('>i2'), SAMPLES * SAMPLES) \
                .reshape((SAMPLES, SAMPLES))

            lat_row = int(round((lat - int(lat)) * (SAMPLES - 1), 0))
            lon_row = int(round((lon - int(lon)) * (SAMPLES - 1), 0))

            return elevations[SAMPLES - 1 - lat_row, lon_row].astype(int)

    def network_analysis(self, sample, service_areas):
        print('> Network analysis at the ' + sample + ' level at ' + str(service_areas) + ' buffer radius')
        start_time = timeit.default_timer()

        if sample == 'lda': sample_gdf = self.LDAs
        elif sample == 'parcel': sample_gdf = self.parcels
        else: sample_gdf = None

        # Load data
        nodes = self.nodes
        edges = self.edges
        print(nodes.head(3))
        print(edges.head(3))
        nodes.index = nodes.osmid

        # Create network
        net = pdna.Network(nodes.geometry.x,
                           nodes.geometry.y,
                           edges["from"],
                           edges["to"],
                           edges[["length"]],
                           twoway=True)
        print(net)
        net.precompute(max(service_areas))

        gdfs = {'lda': self.LDAs,
                'parcel': self.parcels,
                'nodes': self.nodes,
                'links': self.edges}
        cols = {'lda': [],
                'parcel': ["NUMBER_OF_BEDROOMS", "NUMBER_OF_BATHROOMS", "elab_use"],
                'nodes': ["one"],
                'links': ["one"]}

        x, y = sample_gdf.centroid.x, sample_gdf.centroid.y
        sample_gdf["node_ids"] = net.get_node_ids(x.values, y.values)

        buffers = {}
        for key, values in cols.items():
            gdf = gdfs[key]
            x, y = gdf.centroid.x, gdf.centroid.y
            gdf["node_ids"] = net.get_node_ids(x.values, y.values)
            gdf["one"] = 1

            # Try to convert to numeric
            for value in values:
                try: pd.to_numeric(gdf[value])
                except:
                    for item in gdf[value].unique():
                        if value in ['residential', 'retail', 'office', 'civic', 'entertainment']:
                            gdf.loc[gdf[value] == item, item] = 1
                            gdf.loc[gdf[value] != item, item] = 0
                            values.append(item)
                    values.remove(value)

            for value in values:
                print('Processing ' + value + ' column at ' + key + ' GeoDataFrame')
                net.set(node_ids=gdf["node_ids"], variable=gdf[value])

                for radius in service_areas:
                    count = net.aggregate(distance=radius, type="count", decay="flat")
                    sum = net.aggregate(distance=radius, type="sum", decay="flat")
                    ave = net.aggregate(distance=radius, type="ave", decay='flat')

                    sample_gdf[value + '_r' + str(radius) + '_count'] = list(count.loc[sample_gdf["node_ids"]])
                    sample_gdf[value + '_r' + str(radius) + '_sum'] = list(sum.loc[sample_gdf["node_ids"]])
                    sample_gdf[value + '_r' + str(radius) + '_ave'] = list(ave.loc[sample_gdf["node_ids"]])

        elapsed = round((timeit.default_timer() - start_time) / 60, 1)
        sample_gdf.to_file(self.gpkg, layer=sample+'_na', driver='GPKG')
        return print('Network analysis processed in ' + str(elapsed) + ' minutes @ ' + str(datetime.datetime.now()))

    # Spatial analysis
    def set_parameters(self, service_areas, unit='lda', samples=None, max_area=7000000, elab_name='Sunset', bckp=True,
                       layer='Optional GeoPackage layer to analyze', buffer_type='circular'):
        # Load GeoDataFrame and assign layer name for LDA
        if unit == 'lda':
            gdf = self.LDAs.loc[self.LDAs.geometry.area < max_area]
            layer = 'land_dissemination_area'

        # Pre process database for elementslab 1600x1600m 'Sandbox'
        elif unit == 'elab_sandbox':
            self.directory = 'Sandbox/'+elab_name
            self.gpkg = elab_name+'.gpkg'
            if 'PRCLS' in layer:
                nodes_gdf = gpd.read_file(self.gpkg, layer='network_intersections')
                edges_gdf = gpd.read_file(self.gpkg, layer='network_streets')
                cycling_gdf = gpd.read_file(self.gpkg, layer='network_cycling')
                if '2020' in layer:
                    self.nodes = nodes_gdf.loc[nodes_gdf['ctrld2020'] == 1]
                    self.edges = edges_gdf.loc[edges_gdf['new'] == 0]
                    self.cycling = cycling_gdf.loc[cycling_gdf['year'] == '2020-01-01']
                    self.cycling['type'] = self.cycling['type2020']
                    self.cycling.reset_index(inplace=True)
                elif '2050' in layer:
                    self.nodes = nodes_gdf.loc[nodes_gdf['ctrld2050'] == 1]
                    self.edges = edges_gdf
                    self.cycling = cycling_gdf
                    self.cycling['type'] = cycling_gdf['type2050']
            self.parcels = gpd.read_file(self.gpkg, layer=layer)
            self.parcels.crs = {'init': 'epsg:26910'}

            # Reclassify land uses and create bedroom and bathroom columns
            uses = {'residential': ['RS_SF_D', 'RS_SF_A', 'RS_MF_L', 'RS_MF_H'],
                    'retail': ['CM', 'MX'],
                    'civic': ['CV'],
                    'green': ['OS']}
            new_uses = []
            index = list(self.parcels.columns).index("LANDUSE")
            all_prim_uses = [item for sublist in list(uses.values()) for item in sublist]
            for row in self.parcels.iterrows():
                for key, value in uses.items():
                    if row[1]['LANDUSE'] in value:
                        new_uses.append(key)
                if row[1]['LANDUSE'] not in all_prim_uses:
                    new_uses.append(row[1]['LANDUSE'])
            self.parcels['elab_use'] = new_uses
            self.parcels['PRIMARY_ACTUAL_USE'] = self.parcels['LANDUSE']
            self.parcels['NUMBER_OF_BEDROOMS'] = 2
            self.parcels['NUMBER_OF_BATHROOMS'] = 1

            # Define GeoDataFrame
            # gdf = gpd.GeoDataFrame(geometry=self.parcels.unary_union.convex_hull)
            gdf = self.parcels[['OBJECTID', 'geometry']]
            gdf.crs = {'init': 'epsg:26910'}
        else: gdf = None

        c_hull = gdf.geometry.unary_union.convex_hull
        if samples is not None:
            gdf = gdf.sample(samples)
            gdf.sindex()
        self.gdfs = {}
        buffers = {}
        for radius in service_areas:
            buffers[radius] = []

        if buffer_type == 'circular':
            # Buffer polygons for cross-scale data aggregation and output one GeoDataframe for each scale of analysis
            for row in gdf.iterrows():
                geom = row[1].geometry
                for radius in service_areas:
                    lda_buffer = geom.centroid.buffer(radius)
                    buffer_diff = lda_buffer.intersection(c_hull)
                    buffers[radius].append(buffer_diff)

        for radius in service_areas:
            self.gdfs['_r' + str(radius) + 'm'] = gpd.GeoDataFrame(geometry=buffers[radius], crs=gdf.crs)
            sindex = self.gdfs['_r' + str(radius) + 'm'].sindex
        self.params = {'gdf': gdf, 'service_areas': service_areas, 'layer': layer, 'backup': bckp}
        print(self.gdfs)
        print('Parameters set for ' + str(len(self.gdfs)) + ' spatial scales')
        return self.params

    def geomorph_indicators(self):
        # 'Topographical Unevenness'
        gdf = self.params['gdf']
        service_areas = self.params['service_areas']
        dict_of_dicts = {}
        try:
            for radius in service_areas:
                series = gdf.read_file(self.gpkg, layer='land_dissemination_area')[
                    'topo_unev_r' + str(radius) + 'm']
        except:
            start_time = timeit.default_timer()
            print('> Processing topographical unevenness')

            topo_unev = {}
            elevations = {}
            processed_keys = []
            for in_gdf, key in zip(self.gdfs.values(), self.gdfs.keys()):
                topo_unev[key] = []

                for pol, i in zip(in_gdf.geometry, enumerate(in_gdf.geometry)):
                    elevations[key] = []
                    geom_simp = pol.simplify(math.sqrt(pol.area) / 10, preserve_topology=False)
                    try:
                        bound_pts = geom_simp.exterior.coords
                    except:
                        bound_pts = geom_simp[0].exterior.coords
                    for pt in bound_pts:
                        # Reproject point to WGS84
                        pt_gdf = gpd.GeoDataFrame(geometry=[Point(pt)])
                        pt_gdf.crs = {'init': 'epsg:' + self.crs}
                        pt_gdf.to_crs({'init': 'epsg:4326'}, inplace=True)
                        # Define .hgt file to extract topographic data based on latitude and longitude
                        lon = str((int(pt_gdf.geometry.x[0]) * -1) + 1)
                        lat = str(int(pt_gdf.geometry.y[0]))
                        filename = 'N' + lat + 'W' + lon + '.hgt'
                        # Extract elevation data from .hgt file and add it to dictionary
                        elevation = self.elevation(self.directory+'Topography/' + filename,
                                                   lon=pt_gdf.geometry.x[0], lat=pt_gdf.geometry.y[0])
                        elevations[key].append(elevation)
                        elevations[key].sort()
                    print(elevations[key])
                    unev = elevations[key][len(elevations[key]) - 1] - elevations[key][0]
                    for key2 in processed_keys:
                        if topo_unev[key2][i[0]] > unev:
                            unev = topo_unev[key2][i[0]]
                    topo_unev[key].append(unev)
                    print(topo_unev)
                processed_keys.append(key)
            dict_of_dicts['topo_unev'] = topo_unev
            elapsed = round((timeit.default_timer() - start_time) / 60, 1)
            print('Topographical unevenness processed in ' + str(elapsed) + ' minutes')

        for key, value in dict_of_dicts.items():
            for key2, value2 in value.items():
                gdf[key + key2] = value2
        print(gdf)
        copyfile(self.gpkg, self.gpkg+'.bak')
        gdf.to_file(self.gpkg, layer='land_dissemination_area')
        return gdf

    def demographic_indicators(self):
        unit = 'lda'
        # WIP
        census_gdf = gpd.read_file(self.census_file)
        bound_gdf = gpd.read_file(self.gpkg, layer='land_census_subdivision')
        bound_gdf.crs = {'init': 'epsg:3348'}
        bound_gdf.to_crs({'init': 'epsg:4326'}, inplace=True)
        city_lda = census_gdf[census_gdf.within(bound_gdf.geometry[0])]
        print(city_lda)

        # Set up driver for web scraping
        options = Options()
        options.set_preference("browser.download.folderList", 1)
        options.set_preference("browser.download.manager.showWhenStarting", False)
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")
        driver = webdriver.Firefox(executable_path=r'C:\WebDrivers\geckodriver.exe', options=options)
        driver.set_page_load_timeout(1)

        gdf = gpd.read_file('Databases/BC Assessment.gdb')
        print(gdf)

        # Population density (census 2016)
        if unit == 'lda':
            for da, csd in zip(city_lda['DAUID'], city_lda['CSDUID']):
                print(str(csd) + '-' + str(da))
                url = 'https://www12.statcan.gc.ca/census-recensement/2016/dp-pd/prof/details/' \
                      'download-telecharger/current-actuelle.cfm?Lang=E&Geo1=DA&Code1=' + da + '&Geo2=CSD&Code2=' \
                      + csd + '&B1=All&type=0&FILETYPE=CSV'
                print(url)
                driver.get(url)
        driver.close()
        return None

    def density_indicators(self):
        # Process 'Parcel Density', 'Dwelling Density', 'Bedroom Density', 'Bathroom Density', 'Retail Density'
        gdf = self.params['gdf']
        layer = self.params['layer']
        dict_of_dicts = {}

        print('> Processing spatial density indicators')
        start_time = timeit.default_timer()

        # Drop index columns from previous processing
        if 'index_right' in self.parcels.columns:
            self.parcels.drop('index_right', axis=1, inplace=True)
        if 'index_left' in self.parcels.columns:
            self.parcels.drop('index_left', axis=1, inplace=True)

        # Create empty dictionaries and lists
        parc_den = {}
        dwell_den = {}
        bed_den = {}
        bath_den = {}
        dest_den = {}
        dest_ct = {}
        dwell_ct = {}

        for geom, key in zip(self.gdfs.values(), self.gdfs.keys()):
            parc_den[key] = []
            dwell_den[key] = []
            bed_den[key] = []
            bath_den[key] = []
            dest_den[key] = []
            dest_ct[key] = []
            dwell_ct[key] = []

        # Iterate over GeoDataFrames
        for geom, key in zip(self.gdfs.values(), self.gdfs.keys()):

            if 'index_right' in geom.columns:
                geom.drop('index_right', axis=1, inplace=True)
            if 'index_left' in geom.columns:
                geom.drop('index_left', axis=1, inplace=True)

            jgdf = gpd.sjoin(geom, self.parcels, how='right', op="intersects")
            for id in range(len(gdf)):
                fgdf = jgdf.loc[(jgdf['index_left'] == id)]
                if len(fgdf) == 0:
                    parc_den[key].append(0)
                    dwell_den[key].append(0)
                    bed_den[key].append(0)
                    bath_den[key].append(0)
                    dest_den[key].append(0)
                    dwell_ct[key].append(0)
                    dest_ct[key].append(0)

                else:
                    area = geom.loc[id].geometry.area

                    parc_gdf = fgdf.drop_duplicates(subset='geometry')
                    parc_den[key].append(len(parc_gdf)/area)

                    dwell_gdf = fgdf.loc[fgdf['elab_use'] == 'residential']
                    dwell_den[key].append(len(dwell_gdf)/area)
                    dwell_ct[key].append(len(dwell_gdf))

                    bed_den[key].append(dwell_gdf['NUMBER_OF_BEDROOMS'].sum()/area)
                    bath_den[key].append(fgdf['NUMBER_OF_BATHROOMS'].sum()/area)

                    dest_gdf = fgdf.loc[(fgdf['elab_use'] == 'retail') |
                                        (fgdf['elab_use'] == 'office') |
                                        (fgdf['elab_use'] == 'entertainment')]
                    dest_den[key].append(len(dest_gdf)/area)
                    dest_ct[key].append(len(dest_gdf))

        dict_of_dicts['parc_den'] = parc_den
        dict_of_dicts['dwell_ct'] = dwell_ct
        dict_of_dicts['dwell_den'] = dwell_den
        dict_of_dicts['bed_den'] = bed_den
        dict_of_dicts['bath_den'] = bath_den
        dict_of_dicts['dest_ct'] = dest_ct
        dict_of_dicts['dest_den'] = dest_den

        # Append all processed data to a single GeoDataFrame, backup and export
        for key, value in dict_of_dicts.items():
            for key2, value2 in value.items():
                gdf[key + key2] = value2
        if self.params['backup']:
            copyfile(self.gpkg, self.directory+'/ArchiveOSX/'+self.municipality+' - '+str(datetime.date.today())+'.gpkg')
        gdf.to_file(self.gpkg, layer=layer, driver='GPKG')
        elapsed = round((timeit.default_timer() - start_time) / 60, 1)
        return print('Density indicators processed in ' + str(elapsed) + ' minutes @ ' + str(datetime.datetime.now()))

    def diversity_indicators(self):
        # Process 'Land Use Diversity', 'Parcel Size Diversity', 'Dwelling Diversity'
        gdf = self.params['gdf']
        layer = self.params['layer']
        service_areas = self.params['service_areas']
        dict_of_dicts = {}

        print('> Processing spatial diversity indicators')
        start_time = timeit.default_timer()

        # Drop index columns from previous processing
        if 'index_right' in self.parcels.columns:
            self.parcels.drop('index_right', axis=1, inplace=True)
        if 'index_left' in self.parcels.columns:
            self.parcels.drop('index_left', axis=1, inplace=True)

        # Create empty dictionaries and lists
        use_div = {}
        dwell_div = {}
        parc_area_div = {}
        for geom, key in zip(self.gdfs.values(), self.gdfs.keys()):
            use_div[key] = []
            dwell_div[key] = []
            parc_area_div[key] = []

        # Iterate over GeoDataFrames
        for geom, key in zip(self.gdfs.values(), self.gdfs.keys()):

            if 'index_right' in geom.columns:
                geom.drop('index_right', axis=1, inplace=True)
            if 'index_left' in geom.columns:
                geom.drop('index_left', axis=1, inplace=True)

            jgdf = gpd.sjoin(geom, self.parcels, how='right', op="intersects")
            for id in range(len(gdf)):
                fgdf = jgdf.loc[(jgdf['index_left'] == id)]
                if len(fgdf) == 0:
                    use_div[key].append(0)
                    dwell_div[key].append(0)
                    parc_area_div[key].append(0)
                else:
                    use_gdf = fgdf.loc[(fgdf['elab_use'] == 'residential') |
                                       (fgdf['elab_use'] == 'entertainment') |
                                       (fgdf['elab_use'] == 'civic') |
                                       (fgdf['elab_use'] == 'retail') |
                                       (fgdf['elab_use'] == 'office')]
                    use_div[key].append(shannon_div(use_gdf, 'elab_use'))

                    res_gdf = fgdf.loc[(fgdf['elab_use'] == 'residential')]
                    dwell_div[key].append(shannon_div(res_gdf, 'PRIMARY_ACTUAL_USE'))

                    parcel_gdf = fgdf.drop_duplicates(subset=['geometry'])
                    parcel_gdf['area'] = parcel_gdf.geometry.area
                    parcel_gdf.loc[parcel_gdf['area'] < 400, 'area_group'] = '<400'
                    parcel_gdf.loc[(parcel_gdf['area'] > 400) & (parcel_gdf['area'] < 800), 'area_group'] = '400><800'
                    parcel_gdf.loc[(parcel_gdf['area'] > 800) & (parcel_gdf['area'] < 1600), 'area_group'] = '800><1600'
                    parcel_gdf.loc[(parcel_gdf['area'] > 1600) & (parcel_gdf['area'] < 3200), 'area_group'] = '1600><3200'
                    parcel_gdf.loc[(parcel_gdf['area'] > 3200) & (parcel_gdf['area'] < 6400), 'area_group'] = '3200><6400'
                    parcel_gdf.loc[parcel_gdf['area'] > 6400, 'area_group'] = '>6400'
                    parc_area_div[key].append(shannon_div(parcel_gdf, 'area_group'))

        dict_of_dicts['use_div'] = use_div
        dict_of_dicts['dwell_div'] = dwell_div
        dict_of_dicts['parc_area_div'] = parc_area_div

        # Append all processed data to a single GeoDataFrame, backup and export
        for key, value in dict_of_dicts.items():
            for key2, value2 in value.items():
                gdf[key + key2] = value2
        if self.params['backup']:
            copyfile(self.gpkg, self.directory+'/ArchiveOSX/'+self.municipality+' - '+str(datetime.date.today())+'.gpkg')
        gdf.to_file(self.gpkg, layer=layer)
        elapsed = round((timeit.default_timer() - start_time) / 60, 1)
        return print('Diversity indicators processed in ' + str(elapsed) + ' minutes @ ' + str(datetime.datetime.now()))

    def street_network_indicators(self, net_tolerance=10):
        # Define GeoDataframe sample unit
        gdf = self.params['gdf']
        layer = self.params['layer']
        service_areas = self.params['service_areas']
        dict_of_dicts = {}

        # 'Intersection Density', 'Link-node Ratio', 'Network Density', 'Average Street Length'
        start_time = timeit.default_timer()
        print('> Processing general network indicators')
        intrs_den = {}
        linkn_rat = {}
        netw_den = {}
        strt_len = {}
        for geom, key in zip(self.gdfs.values(), self.gdfs.keys()):
            intrs_den[key] = []
            linkn_rat[key] = []
            netw_den[key] = []
            strt_len[key] = []
            exceptions = []
            for pol in geom.geometry:
                nodes_w = self.nodes[self.nodes.geometry.within(pol)]
                try:
                    nodes_w = nodes_w.geometry.buffer(net_tolerance).unary_union
                    len_nodes_w = len(nodes_w)
                    if len(nodes_w) == 0:
                        len_nodes_w = 1
                except:
                    exceptions.append('exception')
                    len_nodes_w = 1
                intrs_den[key].append(round(len_nodes_w / (pol.area / 10000), 2))
                edges_w = self.edges[self.edges.geometry.within(pol)]
                len_edges_w = len(edges_w)
                if len(edges_w) == 0:
                    len_edges_w = 1
                edges_w_geom_length = edges_w.geometry.length
                if len(edges_w_geom_length) == 0:
                    edges_w_geom_length = [1, 1]
                linkn_rat[key].append(round(len_edges_w / len_nodes_w, 2))
                netw_den[key].append(round(sum(edges_w_geom_length) / (pol.area / 10000), 5))
                strt_len[key].append(round(sum(edges_w_geom_length) / len(edges_w_geom_length), 2))
            print('Network iterations at the ' + key + ' scale finished with a total of ' + str(
                  len(exceptions)) + ' exceptions')
        dict_of_dicts['intrs_den'] = intrs_den
        dict_of_dicts['linkn_rat'] = linkn_rat
        dict_of_dicts['netw_den'] = netw_den
        dict_of_dicts['strt_len'] = strt_len
        elapsed = round((timeit.default_timer() - start_time) / 60, 1)
        print('General network indicators processed in ' + str(elapsed) + ' minutes')

        for key, value in dict_of_dicts.items():
            for key2, value2 in value.items():
                gdf[key + key2] = value2
        copyfile(self.gpkg, self.gpkg+'.bak')
        gdf.to_file(self.gpkg, layer=layer)
        print('Processing finished @ ' + str(datetime.datetime.now()))
        return None

    def cycling_network_indicators(self):
        # Read file and pre-process geometry according to its type
        if str(type(self.cycling.geometry[0])) != "<class 'shapely.geometry.polygon.Polygon'>":
            print('> Geometry is not polygon, buffering')
            self.cycling.geometry = self.cycling.buffer(40)

        gdf = self.params['gdf']
        layer = self.params['layer']

        if 'index_left' in gdf.columns:
            gdf.drop(['index_left'], axis=1, inplace=True)
        if 'index_right' in gdf.columns:
            gdf.drop(['index_right'], axis=1, inplace=True)

        dict_of_dicts = {}
        start_time = timeit.default_timer()
        print('> Processing cycling network indicators')

        onstreet = {}
        offstreet = {}
        informal = {}
        all_cycl = {}
        onstreet_gdf = self.cycling[self.cycling['type'] == 'onstreet']
        offstreet_gdf = self.cycling[self.cycling['type'] == 'offstreet']
        informal_gdf = self.cycling[self.cycling['type'] == 'informal']

        for geom, key in zip(self.gdfs.values(), self.gdfs.keys()):
            onstreet[key] = []
            offstreet[key] = []
            informal[key] = []
            all_cycl[key] = []
            for pol in geom.geometry:
                onstreet_w = onstreet_gdf[onstreet_gdf.geometry.within(pol)]
                offstreet_w = offstreet_gdf[offstreet_gdf.geometry.within(pol)]
                informal_w = informal_gdf[informal_gdf.geometry.within(pol)]
                all_cycl_w = gdf[gdf.geometry.within(pol)]
                if len(onstreet_w.geometry) == 0: onstreet[key].append(0)
                else: onstreet[key].append(sum(onstreet_w.geometry.area))
                if len(offstreet_w.geometry) == 0: offstreet[key].append(0)
                else: offstreet[key].append(sum(offstreet_w.geometry.area))
                if len(informal_w.geometry) == 0: informal[key].append(0)
                else: informal[key].append(sum(informal_w.geometry.area))
                if len(all_cycl_w.geometry) == 0: all_cycl[key].append(0)
                else: all_cycl[key].append(sum(all_cycl_w.geometry.area))
        print(all_cycl)

        dict_of_dicts['cycl_onstreet'] = onstreet
        dict_of_dicts['cycl_offstreet'] = offstreet
        dict_of_dicts['cycl_informal'] = informal
        dict_of_dicts['all_cycl'] = all_cycl

        for key, value in dict_of_dicts.items():
            for key2, value2 in value.items():
                gdf[key + key2] = value2
        if self.params['backup']:
            copyfile(self.gpkg, self.directory+'ArchiveOSX/'+self.municipality+' - '+str(datetime.date.today())+'.gpkg')
        gdf.to_file(self.gpkg, layer=layer)

        elapsed = round((timeit.default_timer() - start_time) / 60, 1)
        return print('Cycling network indicators processed in ' + str(elapsed) + ' minutes')

    # Export results
    def export_map(self):
        
        # Process geometry
        boundaries = self.LDAs.geometry.boundary
        centroids = gpd.GeoDataFrame(geometry=self.LDAs.geometry.centroid)
        buffers = {'radius': [], 'geometry': []}
        for radius in self.params['service_areas']:
            for geom in centroids.geometry.buffer(radius):
                buffers['radius'].append(radius)
                buffers['geometry'].append(geom)
        buffers_gdf = gpd.GeoDataFrame(buffers)
        buffer_bounds = gpd.GeoDataFrame(geometry=buffers_gdf['geometry'].boundary)
        buffer_bounds['radius'] = buffers_gdf['radius']

        COLOR_MAP = 'viridis'
        ALPHA = 0.05

        cmap = cm.get_cmap(COLOR_MAP)
        colormap_r = ListedColormap(cmap.colors[::-1])

        # Plot geometry
        fig, ax = plt.subplots(1, 1)
        buffer_bounds.plot(ax=ax, column='radius', colormap=COLOR_MAP, alpha=ALPHA*2)
        boundaries.plot(ax=ax, color='black', linewidth=0.2, linestyle='solid', alpha=0.6)
        centroids.plot(ax=ax, color='#88D2D5', markersize=0.2)
        plt.axis('off')
        plt.savefig('Diagrams/'+self.municipality+' - Mobility Diagram.png', dpi=600)

        return self

    def linear_correlation_lda(self):
        gdf = gpd.read_file(self.gpkg, layer='land_dissemination_area')
        gdf = gdf.loc[gdf.geometry.area < 7000000]
        r = gdf.corr(method='pearson')
        r.to_csv(self.directory + self.municipality + '_r.csv')
        print(r)

    def export_destinations(self):
        dest_gdf = self.parcels.loc[(self.parcels['elab_use'] == 'retail') |
                                    (self.parcels['elab_use'] == 'office') |
                                    (self.parcels['elab_use'] == 'entertainment')]
        dest_gdf['geometry'] = dest_gdf.geometry.centroid
        dest_gdf.drop_duplicates('geometry')
        dest_gdf.to_file(self.directory+'Shapefiles/'+self.municipality+' - Destinations.shp', driver='ESRI Shapefile')
        return self

    def export_parcels(self):
        gdf = self.parcels
        gdf.to_file('Shapefiles/' + self.params['layer'], driver='ESRI Shapefile')
        for col in gdf.columns:
            if str(type(gdf[col][0])) == "<class 'numpy.float64'>" or str(type(gdf[col][0])) == "<class 'numpy.int64'>":
                if sum(gdf[col]) == 0:
                    gdf.drop(col, inplace=True, axis=1)
                    print(col + ' column removed')
        gdf.to_file('Shapefiles/'+self.params['layer']+'_num', driver='ESRI Shapefile')
        return self

    def gravity(self):
        # WIP
        gdf = gpd.read_file(self.gpkg, layer='land_dissemination_area')
        flows = {'origin': [], 'destination': [], 'flow': []}
        for oid in gdf.DAUID:
            for did in gdf.DAUID:
                flows['origin'].append(oid)
                flows['destination'].append(did)
                population = gdf.loc[gdf.DAUID == oid]['pop'].reset_index(drop=True)[0]
                destinations = gdf.loc[gdf.DAUID == did]['dest_ct_lda'].reset_index(drop=True)[0]
                if destinations == 0: destinations = 1
                print(str(oid)+' to '+str(did)+': '+population+' people to '+str(destinations))
                flows['flow'].append(population * destinations)
                print(population * destinations)
        return self

    def export_databases(self):
        layers = ['network_streets', 'land_dissemination_area', 'land_assessment_fabric']
        directory = '/Users/nicholasmartino/Desktop/temp/'
        for layer in layers:
            print('Exporting layer: '+layer)
            gdf = gpd.read_file(self.gpkg, layer=layer)
            gdf.to_file(directory+self.municipality+' - '+layer+'.shp', driver='ESRI Shapefile')
        return self


class Sandbox:
    def __init__(self, name, geodatabase, layers):
        self.name = name
        self.gdb = geodatabase
        self.directory = "../Geospatial/Databases/Sandbox/"
        self.layers = layers

    def morph_indicators(self):
        os.chdir(self.directory + self.name)
        for layer in self.layers:
            model = City()
            model.set_parameters(unit='elab_sandbox', service_areas=[400, 800, 1600], elab_name=self.name, bckp=False,
                                 layer=layer)
            model.density_indicators()
            model.diversity_indicators()
            model.street_network_indicators()
            model.cycling_network_indicators()
            model.export_parcels()
        return self

    def morph_indic_1600(self, radius=400, population_density=True, dwelling_density=True, retail_density=True,
                         dwelling_diversity=True, use_diversity=True, intersection_density=True):
        # Urban design indicators using superpatch as input
        filepath = self.directory + '/' + self.buildings + '_indicators' + str(radius) + '.geojson'
        if os.path.exists(filepath):
            print(filepath + ' Already Exists!')
            return filepath
        else:
            # Calculate livability indexes
            gdf = gpd.read_file(self.directory + '/' + self.buildings)
            sindex = gdf.sindex
            print(sindex)
            gdf.crs = {'init': 'epsg:26910'}
            c_hull = gdf.unary_union.convex_hull
            # Generate 400m buffer
            buffer = gdf.centroid.buffer(radius)
            cl_buffers = buffer.intersection(c_hull)
            den_pop = []
            den_dwe = []
            den_ret = []
            div_use = []
            div_dwe = []
            inters_counts = []
            inters_den = []
            den_routes = []

            for pol, n in zip(cl_buffers, range(len(cl_buffers))):
                intrs = gdf.geometry.map(lambda x: x.intersects(pol))
                filt_gdf = gdf[intrs]
                area = pol.area
                print(filepath + ' - iteration #' + str(n))

                # Intensity
                # Population Density = Number of residents / Buffer area
                if population_density:
                    res_count = filt_gdf.res_count.sum()
                    den_pop.append(res_count / area)
                    print('population_density: done!')

                # Dwelling Density = Number of residential units / Buffer area
                if dwelling_density:
                    try:
                        units = filt_gdf.res_units.sum()
                    except:
                        units = filt_gdf.n_res_unit.sum()
                    den_dwe.append(float(units) / float(area))
                    print('dwelling_density: done!')

                # Retail Density = Footprint area of buildings that have retail in the ground floor / Buffer area
                if retail_density:
                    try:
                        ret_area = filt_gdf[filt_gdf['shell_type'] == 'Retail'].Shape_Area.sum()  # Sunset
                    except:
                        ret_area = filt_gdf[filt_gdf['gr_fl_use'] == 'Retail'].ftprt_area.sum()  # WestBowl
                    try:
                        f_ret_area = filt_gdf[filt_gdf['shell_type'] == 'Food_Retail'].Shape_Area.sum()  # Sunset
                    except:
                        f_ret_area = filt_gdf[filt_gdf['gr_fl_use'] == 'Food_retail'].ftprt_area.sum()  # WestBowl
                    t_ret_area = ret_area + f_ret_area
                    den_ret.append(t_ret_area / area)
                    print('retail_density: done!')

                # Diversity
                # Dwelling Diversity = Shannon diversity index of elementsdb cases
                if dwelling_diversity:
                    div_dwe.append(shannon_div(filt_gdf, 'case_name'))
                    print('dwelling_diversity: done!')

                # Land Use Diversity = Shannon diversity index of land use categories
                if use_diversity:
                    try:
                        div_use.append(shannon_div(filt_gdf, 'LANDUSE'))  # Sunset
                    except:
                        div_use.append(shannon_div(filt_gdf, 'landuse'))  # WestBowl
                    print('use_diversity: done!')

                # Route density
                streets_gdf = gpd.read_file(self.directory + '/' + self.street_net)
                if radius > 600:
                    bike_gdf = streets_gdf[streets_gdf['Bikeways'] == 1]
                    bike_intrs = bike_gdf.geometry.map(lambda x: x.intersects(pol))
                    den_route = sum(bike_gdf[bike_intrs].geometry.length) / area
                else:
                    streets_intrs = streets_gdf.geometry.map(lambda x: x.intersects(pol))
                    den_route = sum(streets_gdf[streets_intrs].geometry.length) / area
                print('route_density: done!')

                # Intersection density
                if intersection_density:
                    cross_gdf = gpd.read_file(self.directory + '/' + self.inters)
                    intersections = cross_gdf[cross_gdf.geometry.map(lambda x: x.intersects(pol))]
                    inters_count = len(intersections)
                    inters_den = inters_count / area
                    print('inters_den: done!')

                d = {'geom': [pol]}
                pol_df = pd.DataFrame(data=d)
                pol_gdf = gpd.GeoDataFrame(pol_df, geometry='geom')
                pol_gdf.crs = {'init': 'epsg:26910'}
                pol_gdf = pol_gdf.to_crs({'init': 'epsg:4326'})

                x_centroid = pol_gdf.geometry.centroid.apply(lambda p: p.x)[0]
                y_centroid = pol_gdf.geometry.centroid.apply(lambda p: p.y)[0]

                bbox = pol_gdf.total_bounds
                x_1 = bbox[0]
                y_1 = bbox[1]
                x_2 = bbox[2]
                y_2 = bbox[3]

                d = {'id': ['centr', 'pt1', 'pt2'], 'lon': [x_centroid, x_1, x_2], 'lat': [y_centroid, y_1, y_2]}
                df = pd.DataFrame(data=d)
                # print(df)

                i = self.buildings
                if '2020' in i:
                    gdf['route_qlty'] = 1
                    inters_counts.append(inters_count)
                elif '2030' in i:
                    gdf['route_qlty'] = 6
                    inters_counts.append(inters_count)
                elif '2040' in i or '2050' in i:
                    gdf['route_qlty'] = 10
                    inters_counts.append(inters_count + 2)

                den_routes.append(den_route)

            # Proximity
            gdf = gdf.replace(9999, 1600)
            gdf['proximity'] = 1000 / (gdf.d2bike + gdf.d2bus + gdf.d2comm + gdf.d2OS + gdf.d2CV)

            # Add columns
            try:
                gdf['experiment'] = i
            except:
                pass
            try:
                gdf['den_pop'] = den_pop
                gdf['den_pop'] = gdf['den_pop'].fillna(0)
            except:
                pass
            try:
                gdf['den_dwe'] = den_dwe
                gdf['den_dwe'] = gdf['den_dwe'].fillna(0)
            except:
                pass
            try:
                gdf['den_ret'] = den_ret
                gdf['den_ret'] = gdf['den_ret'].fillna(0)
            except:
                pass
            try:
                gdf['div_use'] = div_use
                gdf['div_use'] = gdf['div_use'].fillna(0)
            except:
                pass
            try:
                gdf['div_dwe'] = div_dwe
                gdf['div_dwe'] = gdf['div_dwe'].fillna(0)
            except:
                pass
            try:
                gdf['den_route'] = den_routes
                gdf['den_route'] = gdf['den_route'].fillna(0)
            except:
                pass
            try:
                gdf['inters_count'] = inters_counts
                gdf['inters_count'] = gdf['inters_count'].fillna(0)
            except:
                pass
            try:
                gdf['inters_den'] = inters_den
                gdf['inters_den'] = gdf['inters_den'].fillna(0)
            except:
                pass

            # Export
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print('success :)')
                except:
                    print('fail :(')
            gdf.to_file(filepath, driver='GeoJSON')
            return gdf
