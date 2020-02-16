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
import os
import time
import timeit
import warnings

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from shapely.geometry import Point
from Statistics.basic_stats import shannon_div


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
        except:
            pass
        try:
            self.boundary = gpd.read_file(self.gpkg, layer='land_municipal_boundary')
        except:
            pass
        try:
            self.LDAs = gpd.read_file(self.gpkg, layer='land_dissemination_area')
        except:
            pass
        try:
            self.walking_net = gpd.read_file(self.gpkg, layer='network_streets_walking')
            self.cycling_net = gpd.read_file(self.gpkg, layer='network_streets_cycling')
            self.driving_net = gpd.read_file(self.gpkg, layer='network_streets_driving')
        except:
            pass
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=DeprecationWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        warnings.simplefilter(action='ignore', category=ResourceWarning)
        print('Class ' + self.city_name + ' created @ ' + str(datetime.datetime.now()))

    # Scrape and clean data
    def check_file_databases(self, bound=True, net=True, census=True, bcaa=True, icbc=True):
        # Check if boundary data exists and download it from OSM if not
        if bound:
            try:
                self.boundary = gpd.read_file(self.gpkg, layer='land_municipal_boundary')
                print(self.city_name + ' boundary read from database')
            except:
                print('No boundary in database, downloading for ' + self.city_name)
                self.boundary = ox.gdf_from_place(self.municipality)  # gdf
                self.boundary.crs = {'init': 'epsg:4326'}
                self.boundary.to_crs({'init': 'epsg:26910'}, inplace=True)
                self.boundary.to_file(self.gpkg, layer='land_municipal_boundary', driver='GPKG')
                self.boundary = gpd.read_file(self.gpkg, layer='land_municipal_boundary')
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
                edges = gpd.read_file('osm/edges/edges.shp')
                nodes = gpd.read_file('osm/nodes/nodes.shp')
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
                self.LDAs.to_file(self.gpkg, layer='land_dissemination_area')
                print('Census dissemination area joined for ' + self.city_name)

        # Check if BC Assessment data exists and join it from BCAA database if not
        if bcaa:
            try:
                self.parcels = gpd.read_file(self.gpkg, layer='land_assessment_fabric')
                print(self.city_name + ' BC Assessment data read from database')
            except:
                # Spatial join with spatial index BC Assessment data
                start_time = time.time()
                gdf = gpd.read_file(self.assessment, layer='land_assessment_fabric')
                print("BC Assessment data read in %s minutes" % str(
                    round((time.time() - start_time) / 60, 2)))
                start_time = time.time()
                matches = gpd.sjoin(gdf, self.boundary, op='within')
                matches.to_file(self.gpkg, layer='land_assessment_fabric')
                print("Spatial join and export for " + self.city_name + " performed in %s minutes " % str(
                    round((time.time() - start_time) / 60, 2)))
                self.parcels = gpd.read_file(self.gpkg, layer='land_assessment_fabric')
                self.parcels.sindex()

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
        # filepath = 'Databases/BCA/BC Assessment.csv'
        # try: df = pd.read_csv(filepath)
        # except:
        #     # Group data by Property ID
        #     inventory = r'..\Geospatial\Databases\BCA\Inventory Information - RY 2017.csv'
        #     df = pd.read_csv(inventory, engine='python')
        #     df.fillna(0, inplace=True)
        #     df = df.groupby('ROLL_NUM').agg({
        #         'AREA': 'mean',
        #         'JUR': 'first',
        #         'PID': 'first',
        #         'PROPERTY_TYPE': 'first',
        #         'PRIMARY_ACTUAL_USE': 'first',
        #         'PROPERTY_ADDRESS': 'first',
        #         'LEGAL_DESCRIPTION': 'first',
        #         'LAND_TYPE_NAME': 'first',
        #         'LAND_SIZE': 'mean',
        #         'LAND_DEPTH': 'mean',
        #         'ACTUAL_TOTAL': 'mean',
        #         'ACTUAL_LAND': 'mean',
        #         'ACTUAL_IMPR': 'mean',
        #         'REGIONAL_DISTRICT': 'first',
        #         'BLDG_COUNT': 'mean',
        #         'YEAR_BUILT': 'mean',
        #         'TOTAL_FINISHED_AREA': 'mean',
        #         'NUMBER_OF_BATHROOMS': 'sum',
        #         'NUMBER_OF_BEDROOMS': 'sum',
        #         'STRATA_UNIT_AREA': 'mean',
        #         'PREDOMINANT_OCCUPANCY': 'first',
        #         'NUMBER_OF_STOREYS': 'mean',
        #         'GROSS_BUILDING_AREA': 'mean',
        #         'GROSS_LEASABLE_AREA': 'mean',
        #         'NET_LEASABLE_AREA': 'mean'
        #     })
        #     df.to_csv(filepath)
        #     df = pd.read_csv(filepath)
        # print('BCA csv layer loaded with '+str(len(df))+' rows')

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

        try:
            with open(self.directory+'BC_Assessment_unjoined.csv', 'w') as file:
                file.write([set(out_gdf['JUROL']).symmetric_difference(gdf['JUROL'])].sort())
        except: pass
        out_gdf.to_file(self.gpkg, driver='GPKG', layer='land_assessment_fabric')
        return out_gdf

    def aggregate_bca_from_location(self):
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

    def get_census_mobility(self):
        filepath = r'Census\98-400-X2016328_English_CSV_data.csv'

        with open(filepath) as file:
            df = pd.read_csv(file)
            # Drop some useless columns
            to_drop = ['ï»¿"CENSUS_YEAR"',
                       'GEO_LEVEL',
                       'GEO_NAME',
                       'GNR',
                       'DATA_QUALITY_FLAG',
                       'CSD_TYPE_NAME',
                       'ALT_GEO_CODE',
                       'Member ID: Commuting duration (6)',
                       'Notes: Commuting duration (6)',
                       'Member ID: Time leaving for work (7)',
                       'Notes: Time leaving for work (7)',
                       'Member ID: Main mode of commuting (10)',
                       'Notes: Main mode of commuting (10)',
                       'Dim: Distance from home to work (12): Member ID: [1]: Total - Distance from home to work (Note: 2)']
            for col in to_drop:
                try:
                    df.drop(col, axis=1, inplace=True)
                    print(col + ' dropped')
                except:
                    print('drop failed')

            # Drop some useless rows
            to_drop = ['2 or more persons shared the ride to work',
                       'Driver, alone',
                       'Driver, with 1 or more passengers',
                       'Other method',
                       'Passenger, 2 or more persons in the vehicle',
                       'Sustainable transportation']
            df = df[df['DIM: Commuting duration (6)'] == 'Total - Commuting duration']
            df = df[df['DIM: Time leaving for work (7)'] == 'Total - Time leaving for work']
            df = df[df['DIM: Main mode of commuting (10)'] != 'Total - Main mode of commuting']
            for i in to_drop:
                df = df[df['DIM: Main mode of commuting (10)'] != i]

            # Group data and export
            df = df.groupby(['GEO_CODE (POR)', 'DIM: Main mode of commuting (10)']).sum().transpose().stack()
            df = df.transpose().reset_index()
            new_col = []
            for tup in df.columns:
                new_col.append(tup[0] + '_' + tup[1])
            df.columns = new_col
            df.rename(columns={'GEO_CODE (POR)_': 'GEO_CODE'}, inplace=True)
            df['GEO_CODE'] = pd.to_numeric(df['GEO_CODE'])
            df.to_csv(path_or_buf=filepath + '_filtered.csv')

            # Merge df with Census Subdivision boundary file
            gdf = gpd.read_file('Census/lcsd000b16a_e.shp')
            gdf = gdf[gdf['CSDNAME'] == self.city_name]
            gdf.rename(columns={'CSDUID': 'GEO_CODE'}, inplace=True)
            gdf['GEO_CODE'] = pd.to_numeric(gdf['GEO_CODE'])
            df = pd.merge(df, gdf, on='GEO_CODE', how='inner')
            gdf = gpd.GeoDataFrame(df, geometry='geometry')
            gdf.crs = {'init': 'epsg:3348'}

            # Calculate active transport 'market share'
            at_cols = []
            mid_cols = []
            for i in gdf.columns:
                if 'Active transport' in i:
                    at_cols.append(i)
                if 'Member ID' in i:
                    mid_cols.append(i)
            gdf['active_transport'] = gdf[at_cols].sum(axis=1) / gdf[mid_cols].sum(axis=1)
            try:
                gdf.to_file(self.gpkg, layer='land_census_subdivision', driver='GPKG')
                print(gdf)
                print('Census data from ' + self.city_name + ' processed and uploaded :)')
            except:
                print('Failed to upload census ')

        return gdf

    # Spatial analysis
    def set_parameters(self, service_areas, unit='lda', samples=None):
        # Buffer polygons for cross-scale data aggregation and output one GeoDataframe for each scale of analysis
        if unit == 'lda':
            gdf = self.LDAs
        else:
            gdf = None
        if samples is not None:
            gdf = gdf.sample(samples)
            gdf.sindex()

        self.gdfs = {'_' + unit: gdf}
        buffers = {}
        for radius in service_areas:
            buffers[radius] = []
        for row in gdf.iterrows():
            geom = row[1].geometry
            for radius in service_areas:
                lda_buffer = geom.centroid.buffer(radius)
                buffers[radius].append(lda_buffer)
        for radius in service_areas:
            self.gdfs['_r' + str(radius) + 'm'] = gpd.GeoDataFrame(geometry=buffers[radius], crs=gdf.crs)
            sindex = self.gdfs['_r' + str(radius) + 'm'].sindex
        self.params = {'gdf': gdf, 'service_areas': service_areas}
        print(self.gdfs)
        print('Parameters set for ' + str(len(self.gdfs)) + ' spatial scales')
        return self.params

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
            # os.chdir(os.path.dirname(os.path.realpath(__file__)))

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
                        elevation = self.elevation('Databases/Topography/' + filename,
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
        gdf.to_file(self.gpkg, layer='land_dissemination_area')
        return gdf

    def demographic_indicators(self, unit='lda'):
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

        # Retail density
        # Green space density
        # Intersection density
        # Cycling route density

        # Land use diversity
        # Parcel size diversity

        # Topographic unevenness

        # Space syntax indicators

        return None

    def network_indicators(self, net_tolerance=10):
        # Define GeoDataframe sample unit
        gdf = self.params['gdf']
        service_areas = self.params['service_areas']
        dict_of_dicts = {}

        # 'Intersection Density', 'Link-node Ratio', 'Network Density', 'Average Street Length'
        try:
            print([] + 'str')
            # for radius in service_areas:
            #     series1 = gdf.read_file(self.gpkg, layer='land_dissemination_area')['intrs_den_r'+str(radius)+'m']
            #     series2 = gdf.read_file(self.gpkg, layer='land_dissemination_area')['linkn_rat'+str(radius)+'m']
            #     series3 = gdf.read_file(self.gpkg, layer='land_dissemination_area')['netw_den' + str(radius) + 'm']
            #     series4 = gdf.read_file(self.gpkg, layer='land_dissemination_area')['strt_len' + str(radius) + 'm']
        except:
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
        gdf.to_file(self.gpkg, layer='land_dissemination_area')
        print('Processing finished @ ' + str(datetime.datetime.now()))
        return None

    def network_gravity(self):
        return self

    def diversity_indicators(self):
        # Process 'Land Use Diversity', 'Parcel Size Diversity', 'Dwelling Diversity', 'Block Use Mix Dissimilarity'

        gdf = self.params['gdf']
        service_areas = self.params['service_areas']
        dict_of_dicts = {}

        print('> Processing spatial diversity indicators')
        start_time = timeit.default_timer()

        # Reclassify land uses for BC Assessment data
        uses = {
            'residential': ['000 - Single Family Dwelling', '030 - Strata-Lot Residence (Condominium)',
                            '032 - Residential Dwelling with Suite',
                            '033 - Duplex, Non-Strata Side by Side or Front / Back',
                            '034 - Duplex, Non-Strata Up / Down', '035 - Duplex, Strata Side by Side',
                            '036 - Duplex, Strata Front / Back', '039 - Row Housing (Single Unit Ownership)',
                            '041 - Duplex, Strata Up / Down', '047 - Triplex', '050 - Multi-Family (Apartment Block)',
                            '052 - Multi-Family (Garden Apartment & Row Housing)', '053 - Multi-Family (Conversion)',
                            '054 - Multi-Family (High-Rise)', '055 - Multi-Family (Minimal Commercial)',
                            '056 - Multi-Family (Residential Hotel)', '057 - Stratified Rental Townhouse',
                            '058 - Stratified Rental Apartment (Frame Construction)',
                            '059 - Stratified Rental Apartment (Hi-Rise Construction)',
                            '060 - 2 Acres Or More (Single Family Dwelling, Duplex)', '285 - Seniors Licensed Care',
                            '286 - Seniors Independent & Assisted Living'],
            'vacant': ['001 - Vacant Residential Less Than 2 Acres', '051 - Multi-Family (Vacant)', '201 - Vacant IC&I',
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
            'entertainment': ['250 - Theatre Buildings', '254 - Neighbourhood Pub', '256 - Restaurant Only',
                              '266 - Bowling Alley', '270 - Hall (Community, Lodge, Club, Etc.)',
                              '280 - Marine Facilities (Marina)',
                              '600 - Recreational & Cultural Buildings (Includes Curling',
                              '610 - Parks & Playing Fields', '612 - Golf Courses (Includes Public & Private)',
                              '654 - Recreational Clubs, Ski Hills'],
            'institutional': ['210 - Bank', '620 - Government Buildings (Includes Courthouse, Post Office',
                              '625 - Garbage Dumps, Sanitary Fills, Sewer Lagoons, Etc.', '630 - Works Yards',
                              '634 - Government Research Centres (Includes Nurseries &',
                              '640 - Hospitals (Nursing Homes Refer To Commercial Section).',
                              '642 - Cemeteries (Includes Public Or Private).',
                              '650 - Schools & Universities, College Or Technical Schools',
                              '652 - Churches & Bible Schools'],
            'agriculture': ['120 - Vegetable & Truck', '170 - Poultry', '180 - Mixed'],
            'office': ['203 - Stores And/Or Offices With Apartments', '204 - Store(S) And Offices',
                       '208 - Office Building (Primary Use)'],
            'industrial': ['217 - Air Space Title', '272 - Storage & Warehousing (Open)',
                           '273 - Storage & Warehousing (Closed)', '274 - Storage & Warehousing (Cold)',
                           '275 - Self Storage', '276 - Lumber Yard Or Building Supplies', '400 - Fruit & Vegetable',
                           '401 - Industrial (Vacant)', '402 - Meat & Poultry', '403 - Sea Food',
                           '404 - Dairy Products', '405 - Bakery & Biscuit Manufacturing',
                           '406 - Confectionery Manufacturing & Sugar Processing', '408 - Brewery',
                           '414 - Miscellaneous (Food Processing)', '419 - Sash & Door',
                           '423 - IC&I Water Lot (Improved)',
                           '425 - Paper Box, Paper Bag, And Other Paper Remanufacturing.', '428 - Improved',
                           '429 - Miscellaneous (Forest And Allied Industry)',
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
                           '530 - Telecommunications (Other Than Telephone)', '560 - Water Distribution Systems',
                           '580 - Electrical Power Systems (Including Non-Utility']
        }

        new_uses = []
        print(self.parcels.columns)
        for row in self.parcels.iterrows():
            print(row)
            for key in uses.keys():
                for prim_use in uses[key]:
                    if self.parcels.iloc[row, 'PRIMARY ACTUAL USE'] == prim_use:
                        new_uses.append(key)
        self.parcels['elab_use'] = new_uses
        print(self.parcels.elab_use)

        # Process diversity indicators
        use_div = {}
        dwell_div = {}
        parc_area_div = {}
        blck_area_div = {}
        for geom, key in zip(self.gdfs.values(), self.gdfs.keys()):
            use_div[key] = []
            dwell_div[key] = []
            parc_area_div[key] = []
            blck_area_div[key] = []

            jgdf = gpd.sjoin(geom, self.parcels, op="within")



        gdf = self.params['gdf']
        service_areas = self.params['service_areas']

        # Create new columns in the GeoDataframe
        for radius in service_areas:
            print(gdf['use_div_' + str(radius)])



        # Append all processed data to a GeoDataframe
        for key, value in dict_of_dicts.items():
            for key2, value2 in value.items():
                gdf[key + key2] = value2
        print(gdf)
        gdf.to_file(self.gpkg, layer='land_dissemination_area')
        print('Processing finished @ ' + str(datetime.datetime.now()))

        elapsed = round((timeit.default_timer() - start_time) / 60, 1)
        print('General network indicators processed in ' + str(elapsed) + ' minutes')

        return

    def linear_correlation_lda(self):
        gdf = gpd.read_file(self.gpkg, layer='land_dissemination_area')
        r2 = gdf.corr(method='pearson')
        r2.to_csv(self.directory + self.municipality + '_r2.csv')
        print(r2)


class Sandbox:
    def __init__(self, geodatabase):
        self.gdb = geodatabase

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
