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

import gc
import io
import math
import time
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from shapely import affinity
from shapely import ops
from shapely.geometry import Point, LineString, Polygon





class Buildings:
    def __init__(self, gdf, group_by=None, gb_func=None, crs=26910, to_crs=None):
        gdf = gdf.reset_index()

        # Set/Adjust coordinate reference system
        if gdf.crs is None: gdf.crs = crs
        if to_crs is not None: gdf = gdf.to_crs(to_crs)

        if group_by is not None:
            if gb_func is not None:
                gdf = gdf.groupby(group_by, as_index=False).agg(gb_func)

            gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=to_crs)
            gdf = gdf.dissolve(by=group_by)

        # For City of Vancouver open data:
        if 'topelev_m' in gdf.columns: gdf['height'] = gdf['topelev_m'] - gdf['baseelev_m']

        # Get polygon-specific measures
        gdf = Shape(gdf).process()

        self.gdf = gdf
        self.crs = crs
        print("Buildings object created")
        return

    def convex_hull(self):
        gdf = self.gdf.copy()
        gdf['convex_hull'] = [geom.convex_hull for geom in gdf['geometry']]
        gdf['conv_area'] = [geom.area for geom in gdf['convex_hull']]
        gdf['conv_perim'] = [geom.length for geom in gdf['convex_hull']]

        return gdf.drop('convex_hull', axis=1)

    def bounding_box(self):
        gdf = self.gdf.copy()

        # Area and perimeter
        gdf['bound_box'] = [geom.minimum_rotated_rectangle for geom in gdf['geometry']]
        gdf['box_area'] = [geom.area for geom in gdf['bound_box']]
        gdf['box_perim'] = [geom.length for geom in gdf['bound_box']]

        if 'ftprt_perim' in gdf.columns: gdf['ftprt_compactness'] = gdf['ftprt_perim']/(4 * gdf['box_area'])

        # Dimensions
        side_a = [Point(geom.bounds[0], geom.bounds[1]).distance(Point(geom.bounds[2], geom.bounds[1])) for geom in gdf['bound_box']]
        side_b = [Point(geom.bounds[0], geom.bounds[1]).distance(Point(geom.bounds[0], geom.bounds[3])) for geom in gdf['bound_box']]
        gdf['box_width'] = [min(i, j) for i, j in zip(side_a, side_b)]
        gdf['box_length'] = [max(i, j) for i, j in zip(side_a, side_b)]
        gdf['box_w2l_ratio'] = gdf['box_width']/gdf['box_length']

        # Ratios
        gdf['r_boxarea'] = gdf['area']/gdf['box_area']
        gdf['r_boxperim'] = gdf['perimeter']/gdf['box_perim']

        return gdf.drop('bound_box', axis=1)

    def triangulate(self):
        gdf = self.gdf.copy()

        gdf['triangulation'] = [ops.triangulate(geom) for geom in gdf['geometry']]

        gdf['tri_n_triangles'] = [len(list_geom) for list_geom in gdf['triangulation']]
        gdf['tri_area_sum'] = [sum([geom.area for geom in list_geom]) for list_geom in gdf['triangulation']]
        gdf['tri_perim_sum'] = [sum([geom.length for geom in list_geom]) for list_geom in gdf['triangulation']]
        gdf['tri_range'] = [max([geom.length for geom in list_geom]) - min([geom.length for geom in list_geom]) for list_geom in gdf['triangulation']]

        return gdf.drop('triangulation', axis=1)

    def centroid(self):
        gdf = self.gdf.copy()

        gdf['centroid'] = [geom.centroid for geom in gdf['geometry']]

        variance = []
        mean_dists = []
        for geom, centroid in zip(gdf['geometry'], gdf['centroid']):
            v_distances = [Point(coord).distance(centroid) for coord in geom.exterior.coords]
            mean = sum(v_distances)/len(v_distances)
            mean_dists.append(mean)
            dist_mean_sqd = [pow((v_dst - mean), 2) for v_dst in v_distances]
            variance.append(sum(dist_mean_sqd) / (len(v_distances) -1))

        gdf['cnt2vrt_mean'] = mean_dists
        gdf['cnt2vrt_var'] = variance

        return gdf.drop('centroid', axis=1)

    def encl_circle(self):
        gdf = self.gdf.copy()
        return gdf

    def skeleton(self):
        skeletons = Skeleton(gdf=self.gdf, crs=self.crs)
        return skeletons

    def all(self):
        print("> Calculating all indicators for Buildings")
        self.gdf = self.convex_hull()
        self.gdf = self.bounding_box()
        self.gdf = self.triangulate()
        self.gdf = self.centroid()
        self.gdf = self.encl_circle()
        return gpd.GeoDataFrame(self.gdf, crs=self.crs)


class Parcels:
    def __init__(self, gdf, buildings, crs=26910):
        # Clean self-intersecting polygons
        print("Cleaning self-intersecting geometries")
        gdf['geometry'] = [geom if geom.is_valid else geom.buffer(0) for geom in gdf['geometry']]
        gdf = gdf.set_geometry('geometry')

        # Create parcels ids
        gdf['id'] = gdf.index

        # Join parcel id to buildings
        gdf.crs = crs
        buildings.gdf['polygon'] = buildings.gdf['geometry']
        buildings.gdf['geometry'] = buildings.gdf.centroid
        start_time = time.time()
        print(f"Joining information from buildings {buildings.gdf.sindex} to parcels {gdf.sindex}")
        buildings.gdf['parcel_id'] = gpd.sjoin(buildings.gdf, gdf.loc[:, ['geometry']], how="left", op="within").groupby(by='id')['index_right'].first()
        buildings.gdf['geometry'] = buildings.gdf['polygon']
        print(f"Data from buildings joined to parcels in {(time.time() - start_time)/60} minutes")

        # Get polygon-specific measures
        gdf = Shape(gdf).process()

        self.gdf = gdf
        self.buildings = buildings
        print ("Parcels object created")
        return

    def shape(self):
        gdf = self.gdf.copy()
        gdf = Buildings(gdf).all()
        return gdf

    def occupation (self):
        gdf = self.gdf.copy()
        buildings = self.buildings.gdf.copy()

        for i in gdf['id']:
            parcel_buildings = buildings[buildings['parcel_id'] == i]

            if len(parcel_buildings) > 0:
                gdf.at[i, 'coverage'] = (parcel_buildings['area'].sum()/gdf[gdf['id'] == i].area).values

        return gdf

    def plot_boundaries(self):


        return


class Blocks:
    def __init__(self, gdf, parcels, streets, crs=26910):

        gdf = gdf.dropna().reset_index()
        gdf.crs = crs

        # Trim dissolved blocks with streets
        if streets is not None:
            width = 2
            streets.gdf['geometry'] = streets.gdf.buffer(width)
            print(streets.gdf.sindex)
            gdf = gpd.overlay(gdf, streets.gdf, how='difference').reset_index()

        # Create block id and join to buildings and parcels
        gdf['id'] = gdf.index
        parcels.gdf['block_id'] = gpd.sjoin(parcels.gdf, gdf.loc[:, ['geometry']], how="left").groupby(by='id')['index_right'].first()
        parcels.buildings.gdf['block_id'] = gpd.sjoin(parcels.buildings.gdf, gdf.loc[:, ['geometry']], how="left").groupby(by='id')['index_right'].first()

        for i in gdf['id']:
            gdf.loc[gdf['id'] == i, 'n_parcels'] = len(parcels.gdf[parcels.gdf['block_id'] == i])
            gdf.loc[gdf['id'] == i, 'n_buildings'] = len(parcels.buildings.gdf[parcels.buildings.gdf['block_id'] == i])

        self.gdf = gdf
        self.parcels = parcels
        self.crs = crs
        print("Blocks object created")
        return

    def dimension(self):
        gdf = self.gdf.copy()

        gdf['area'] = [geom.area for geom in gdf['geometry']]
        gdf['perimeter'] = [geom.length for geom in gdf['geometry']]

        return gdf


class Streets:
    def __init__(self, gdf, buildings, crs=26910, widths=None, trees=None):
        self.gdf = gdf.to_crs(crs)
        self.barriers = buildings.gdf
        self.trees = trees
        self.widths = widths
        self.crs = crs
        print("Streets object created")
        return

    def dimension(self):
        gdf = self.gdf.copy().reset_index()

        print("> Cleaning street widths")
        if self.widths is not None:
            widths = gpd.read_file(self.widths)
            widths = widths.to_crs(self.crs)

            # Clean data from City of Vancouver open data catalogue
            for i in widths.index:
                if '(m)' in str(widths.at[i, 'width']):
                    try: widths.at[i, 'width'] = float(widths.loc[i, 'width'].split('(m)')[0])
                    except: widths = widths.drop(i)
                elif 'D.L.' in str(widths.at[i, 'width']):
                    widths.at[i, 'width'] = float(widths.loc[i, 'width'].split('D.L.')[1])
                elif '±' in str(widths.at[i, 'width']):
                    widths.at[i, 'width'] = float(widths.loc[i, 'width'].split('±')[0])
                elif 'm' in str(widths.at[i, 'width']):
                    try: widths.at[i, 'width'] = float(widths.loc[i, 'width'].split('m')[0])
                    except: widths = widths.drop(i)
                elif 'R' in str(widths.at[i, 'width']):
                    try: widths.at[i, 'width'] = float(widths.loc[i, 'width'].split('R')[0])
                    except: widths = widths.drop(i)
                elif 'M' in str(widths.at[i, 'width']):
                    widths.at[i, 'width'] = float(widths.loc[i, 'width'].split('M')[0])
                elif widths.at[i, 'width'] in ['-', '+', 'CHD.', 'ST', 'ST.', None]: widths = widths.drop(i)
                else: widths.at[i, 'width'] = float(widths.loc[i, 'width'])

            # Buffer street segments based on average street width
            widths = widths[widths['width'] < 100]
            widths.crs = self.crs
            widths.to_file(f'{directory}/row.shp', driver='ESRI Shapefile')
            widths['geometry'] = widths.buffer(10)

            gdf['id'] = gdf.index
            joined = gpd.sjoin(gdf, widths, how='left')
            joined['width'] = pd.to_numeric(joined['width'])
            joined = joined.groupby('id', as_index=False).mean()
            joined = pd.merge(gdf, joined, on='id', copy=False)
            joined['geometry'] = list(gdf.loc[gdf['id'].isin(joined['id'])]['geometry'])

            # Replace NaN values
            print(f'Width information from {joined["width"].isna().sum()} features could not be joined')
            for use in joined['streetuse'].unique():
                joined.loc[(joined['streetuse'] == use) & (joined['width'].isna()), 'width'] = joined[joined['streetuse'] == use]['width'].mean()
        else: print('Widths layer not found!')

        joined['length'] = [geom.length for geom in joined['geometry']]
        return joined

    def direction(self):
        gdf = self.gdf.copy()

        # Calculate street segment in relation to shortest line from start to end of segment
        print("> Calculating shortest path")
        shortest = []
        for i, segment in zip(gdf.index, gdf['geometry']):
            if segment.__class__.__name__ == 'LineString':
                shortest.append(LineString([Point(segment.coords[0]), Point(segment.coords[1])]).length)
            elif segment.__class__.__name__ == 'MultiLineString':
                shortest.append(LineString([Point(segment[0].coords[0]), Point(segment[0].coords[1])]).length)
                gdf.at[i, 'geometry'] = segment[0]

        gdf['shortest'] = shortest

        # Calculate straightness
        print("> Calculating straightness")
        gdf['straight'] = gdf['shortest']/gdf['length']

        # Calculate azimuth
        print("> Calculating azimuth")
        gdf['azimuth'] = [math.degrees(math.atan2((ln.xy[0][1] - ln.xy[0][0]), (ln.xy[1][1] - ln.xy[1][0]))) for ln in
                            gdf.geometry]

        # Get one way streets from OpenStreetMap

        return gdf

    def connectivity(self):
        gdf = self.gdf.copy().reset_index()
        gdf['id'] = gdf.index
        buffer = gpd.GeoDataFrame(gdf.buffer(1), columns=['geometry'], crs=26910)
        buffer.to_file(f'{directory}/street_segs_buffer.geojson', drive='GeoJSON')

        print(f"> Calculating connections between {buffer.sindex} and {gdf.sindex}")
        overlay = gpd.overlay(gdf, buffer, how='intersection')
        gdf['conn'] = [len(overlay[overlay['id'] == i]) for i in gdf['id'].unique()]
        gdf['deadend'] = [1 if gdf.loc[i, 'conn'] < 2 else 0 for i in gdf.index]
        return gdf

    def visibility(self):
        gdf = self.gdf.copy()
        c_gdf = self.gdf.copy()
        c_gdf.crs = self.crs

        print("> Generating isovists")
        c_gdf['geometry'] = c_gdf.centroid
        gdf['isovists'] = Isovist(origins=c_gdf, barriers=self.barriers).create()
        gdf['iso_area'] = [geom.area for geom in gdf['isovists']]
        gdf['iso_perim'] = [geom.length for geom in gdf['isovists']]
        gdf.drop(['geometry'], axis=1).set_geometry('isovists').to_file(f'{directory}/isovists.geojson', driver='GeoJSON')
        return gdf.drop(['isovists'])

    def greenery(self):
        gdf = self.gdf.copy()
        if self.trees is not None:
            trees = gpd.read_file(self.trees)
            trees = trees.dropna(subset=['geometry'])
            trees.crs = self.crs
            print(trees.sindex)

            # Buffer streets geometries
            if 'width' not in gdf.columns:
                buffer = gpd.GeoDataFrame(gdf.buffer(10), columns=['geometry'], crs=26910)
            else:
                buffer = gpd.GeoDataFrame([g.buffer(w/2) for g, w in zip(gdf['geometry'], gdf['width'])], columns=['geometry'], crs=26910)
            buffer['id'] = buffer.index
            buffer.crs = self.crs
            print(buffer.sindex)

            # Overlay trees and street buffers
            overlay = gpd.overlay(trees, buffer, how='intersection')
            gdf['n_trees'] = [len(overlay[overlay['id'] == i]) for i in buffer['id'].unique()]
            gdf.crs = self.crs
        else: print("Trees layer not found!")
        return gdf

    def all(self):
        print("> Calculating all indicators for Streets")
        # self.gdf = self.dimension()
        # self.gdf = self.direction()
        # self.gdf = self.connectivity()

        try:
            iso_gdf = gpd.read_file(f'{directory}/isovists.geojson').loc[:, ['iso_area', 'iso_perim']]
            iso_gdf['id'] = iso_gdf.index
            self.gdf['id'] = self.gdf.index
            self.gdf = pd.merge(self.gdf, iso_gdf, how="left", copy=False, on='id')
        except:
            print("> Isovists not found")
            self.gdf = self.visibility()

        self.gdf = self.greenery()
        self.gdf.crs = self.crs
        return self.gdf


class Neighborhood:
    def __init__(self):
        print("Neighborhood object created")
        return

    def interconnect(self, min_area=4000, max_cont=1.15):
        return


if __name__ == '__main__':

    # Download BC footprints from StatCan
    zipfile.ZipFile(
        io.BytesIO(requests.get(
            'https://www150.statcan.gc.ca/n1/fr/pub/34-26-0001/2018001/ODB_v2_BritishColumbia.zip?st=qdcH3z04').content)
    ).extractall('tmp/')
    print("Footprint data downloaded")

    ### Buildings ###
    building_gdf = gpd.read_file(filename='tmp/ODB_BritishColumbia/odb_britishcolumbia.shp')
    building_gdf.crs = 3347
    building_gdf = building_gdf.to_crs(26910)
    bld = Buildings(
        gdf=building_gdf, to_crs=26910, group_by='Build_ID',
        # City of Vancouver:
        # gdf=gpd.read_file('https://opendata.vancouver.ca/explore/dataset/building-footprints-2009/download/?format=geojson&timezone=America/Los_Angeles&lang=en&epsg=26910'),# gpd.read_file(f'{directory}/building-footprints-2009.geojson'),
        # gb_func={
        #     'rooftype': 'max',
        #     'baseelev_m': 'min',
        #     'topelev_m': 'max',
        #     'maxht_m': 'max',
        #     'med_slope':'mean',
        #     'geometry': 'first'
        # }
    )

    # Download parcels from BC government open data
    zipfile.ZipFile(
        io.BytesIO(requests.get(
            'https://pub.data.gov.bc.ca/datasets/4cf233c2-f020-4f7a-9b87-1923252fbc24/pmbc_parcel_fabric_poly_svw.zip').content)
    ).extractall('tmp/')
    print("Parcel data downloaded")

    ### Parcels ###
    parcel_gdf = gpd.read_file(filename='tmp/pmbc_parcel_fabric_poly_svw.gdb', layer='pmbc_parcel_fabric_poly_svw')
    parcel_gdf.crs = 3005
    parcel_gdf = parcel_gdf.to_crs(26910)
    pcl = Parcels(
        gdf=parcel_gdf,
        buildings=bld
        # City of Vancouver:
        # gdf=gpd.read_file('https://opendata.vancouver.ca/explore/dataset/property-parcel-polygons/download/?format=geojson&timezone=America/Los_Angeles&lang=en&epsg=26910'),
    )
    pcl.gdf = pcl.occupation()
    pcl.gdf = pcl.gdf[(pcl.gdf.area > 2000) & (pcl.gdf.coverage > 0.2) & (pcl.gdf.elongation < 0.2)]

    # Plot parcel boundaries and building footprints
    pcl_boundary = pcl.gdf.copy()
    pcl_boundary['geometry'] = [geom for geom in pcl_boundary.boundary]
    pcl_boundary = pcl_boundary.set_geometry('geometry')

    # Filter parcels with buildings overlapping its boundary
    bld.gdf = gpd.overlay(bld.gdf, pcl.gdf, how='intersection')

    # Make a convex hull around largest parcel that will be plotted along with all other parcels to standardize the scale
    largest = bld.gdf.sort_values('area_1', ignore_index=True, ascending=False).iloc[0]['geometry'].convex_hull
    largest_centroid = largest.centroid

    print("Plotting parcels and footprint skeletons")
    parcel_ids = pcl.gdf.id
    for k, (j, t) in enumerate(zip(parcel_ids, tqdm(range(len(parcel_ids))))):

        # Move convex hull to parcel to standardize the plot scale
        p_centroid = pcl.gdf[pcl.gdf['id'] == j].centroid
        largest_overlap = affinity.translate(largest, (p_centroid.x - largest_centroid.x).values, (p_centroid.y - largest_centroid.y).values)
        moved = gpd.GeoDataFrame({'geometry': [largest_overlap]}, geometry='geometry')

        j = int(j)

        # Filter buildings with this parcel id
        footprints = bld.gdf[bld.gdf['parcel_id'] == j]

        if len(footprints) > 0:

            if k < (0.8 * len(parcel_ids)):
                # Plot footprint, boundary and parcel
                fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

                moved.plot(ax=ax[0], color='white')
                moved.plot(ax=ax[1], color='white')

                pcl.gdf[pcl.gdf['id'] == j].plot(ax=ax[0], color='green')
                pcl.gdf[pcl.gdf['id'] == j].plot(ax=ax[1], color='green')

                pcl_boundary[pcl_boundary['id'] == j].plot(ax=ax[0], color='red')
                pcl_boundary[pcl_boundary['id'] == j].plot(ax=ax[1], color='red')

                footprints.plot(ax=ax[0], color='blue')

                ax[0].set_axis_off()
                ax[1].set_axis_off()
                fig.savefig(f'../pix2pix/data/footprints/train/{j}.jpg', dpi=256)
                plt.close()

            else:
                # Plot footprint, boundary and parcel
                fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

                moved.plot(ax=ax[0], color='white')
                moved.plot(ax=ax[1], color='white')

                pcl.gdf[pcl.gdf['id'] == j].plot(ax=ax[1], color='green')
                pcl.gdf[pcl.gdf['id'] == j].plot(ax=ax[0], color='green')

                pcl_boundary[pcl_boundary['id'] == j].plot(ax=ax[1], color='red')
                pcl_boundary[pcl_boundary['id'] == j].plot(ax=ax[0], color='red')

                footprints.plot(ax=ax[0], color='blue')

                ax[0].set_axis_off()
                ax[1].set_axis_off()

                fig.savefig(f'../pix2pix/data/footprints/val/{j}.jpg', dpi=256)
                plt.close()

    ### City of Vancouver project indicators ###
    directory = '/Volumes/ELabs/50_projects/20_City_o_Vancouver/SSHRC Partnership Engage/Data/Shapefiles'
    gbd = GeoBoundary('Vancouver, British Columbia')

    ### Streets ###
    stt = Streets(
        gdf=gpd.read_file(f'{directory}/street_row/street_row.shp'),
        widths=f'{directory}/right-of-way-widths.shp',
        trees='https://opendata.vancouver.ca/explore/dataset/street-trees/download/?format=geojson&timezone=America/Los_Angeles&lang=en&epsg=26910',
        buildings=bld
    )

    ### Blocks ###
    blocks = Blocks(
        gdf=gpd.read_file('https://opendata.vancouver.ca/explore/dataset/block-outlines/download/?format=geojson&timezone=America/Los_Angeles&lang=en&epsg=26910'),
        parcels=pcl,
        streets=stt
    )

    ### Calculate indicators ###
    blocks.gdf = blocks.dimension()
    blocks.gdf.to_file(f'{directory}/Blocks.shp', driver='ESRI Shapefile')
    print("Blocks indicators calculated and exported")
    # public = gpd.read_file('https://opendata.vancouver.ca/explore/dataset/public-streets/download/?format=geojson&timezone=America/Los_Angeles&lang=en')
    # lanes = gpd.read_file('https://opendata.vancouver.ca/explore/dataset/lanes/download/?format=geojson&timezone=America/Los_Angeles&lang=en')
    # lanes['streetuse'] = "Lane"
    # gdf = pd.concat([public, lanes]).to_crs(26910)

    pcl.gdf = pcl.shape_indicators()
    pcl.gdf.to_file(f'{directory}/Parcels.shp', driver='ESRI Shapefile')
    print("Parcels indicators calculated and exported")

    pcl.buildings.gdf = pcl.buildings.all()
    pcl.buildings.gdf.to_file(f'{directory}/Buildings.shp', driver='ESRI Shapefile')
    print("Buildings indicators calculated and exported")

    stt.gdf = stt.all()
    stt.gdf = stt.gdf.drop(['std_street', 'from_hundred_block', 'hblock', 'id', 'index_right', 'width_m'], axis=1)
    stt.gdf = stt.gdf.loc[stt.gdf['streetuse'].isin([i for i in stt.gdf['streetuse'].unique() if i not in ['Closed', 'Leased', 'Recreational']])]
    stt.gdf.to_file(f'{directory}/Streets.shp', driver='ESRI Shapefile')
    print("Streets indicators calculated and exported")
