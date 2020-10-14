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

import math

import geopandas as gpd
import pandas as pd
from Analyst import GeoBoundary
from shapely import ops
from shapely.geometry import Point, LineString, Polygon


class Isovist:
    def __init__(self, origin, barriers, radius=500):
        self.origin = origin
        self.barriers = barriers
        self.barriers_index = barriers.sindex
        self.radius = radius

        return

    def create(self, tolerance=100):

        # Buffer origin point according to radius
        buffer = self.origin.buffer(self.radius).simplify(tolerance=tolerance)

        # Create view lines crossing over barriers
        lines = gpd.GeoDataFrame(
            {'geometry':[LineString([self.origin, Point(pt_cds)]) for pt_cds in buffer.boundary.coords]}, geometry='geometry'
        )
        print(lines.sindex)

        # Filter barriers using spatial index
        barriers = self.barriers.iloc[list(self.barriers_index.intersection(lines.total_bounds))]
        print(barriers.sindex)

        # Remove barriers from view lines
        lines_diff = gpd.overlay(lines, barriers, how='difference')

        # Extract lines that intersects with origin
        lines_orig = []
        for geom in lines_diff['geometry']:
            if (geom.__class__.__name__ == 'LineString') and (geom.distance(self.origin) == 0):
                lines_orig.append(geom)
            elif geom.__class__.__name__ == 'MultiLineString':
                for line in geom:
                    if line.distance(self.origin) == 0:
                        lines_orig.append(line)

        # Create isovist polygon
        isovist = Polygon([Point(line.coords[0]) for line in lines_orig])

        return isovist


class Buildings:
    def __init__(self, gdf, group_by=None, gb_func=None, to_crs=None):
        self.gdf = gdf.copy()
        self.gdf['area'] = self.gdf['geometry'].area
        self.gdf = self.gdf.sort_values('area', ascending=False)

        if to_crs is not None:
            self.gdf = self.gdf.to_crs(to_crs)

        if group_by is not None:
            if gb_func is not None:
                self.gdf = self.gdf.groupby(group_by, as_index=False).agg(gb_func)

        self.gdf['area'] = [geom.area for geom in self.gdf['geometry']]
        self.gdf['perimeter'] = [geom.length for geom in self.gdf['geometry']]
        self.gdf['n_vertices'] = [len(geom.exterior.coords) for geom in self.gdf['geometry']]

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

    def all(self):
        self.gdf = self.convex_hull()
        self.gdf = self.bounding_box()
        self.gdf = self.triangulate()
        self.gdf = self.centroid()
        self.gdf = self.encl_circle()
        return self.gdf


class Streets:
    def __init__(self, gdf, crs=26910, widths=None, trees=None):
        self.gdf = gdf.to_crs(crs)
        self.barriers = gpd.read_file(f'{directory}/building-footprints-2009.geojson').to_crs(crs)
        self.trees = trees
        self.widths = widths
        return

    def dimension(self):
        gdf = self.gdf.copy()

        print("> Cleaning street widths")
        if self.widths is not None:
            widths = gpd.read_file(self.widths)

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
            ave_width = widths['width'].mean()
            buf_seg = gpd.GeoDataFrame({
                'id': [i for i in gdf.index],
                'geometry': gdf.buffer(ave_width)
            }, geometry='geometry')
            gdf['id'] = gdf.index
            gdf = pd.merge(gdf, gpd.sjoin(buf_seg, widths, how='left'), on='id', copy=False)
            gdf['geometry'] = gdf['geometry_x']
            gdf = gdf.drop_duplicates(subset=['geometry'])
            if 'width_y' in gdf.columns:
                gdf['width'] = gdf['width_y']
                gdf = gdf.drop('width_y', axis=1)
            gdf = gdf.drop(['geometry_x', 'geometry_y'], axis=1)
        else: print('Widths layer not found!')

        gdf['length'] = [geom.length for geom in gdf['geometry']]
        return gdf

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

        print("> Calculating straightness")
        gdf['straight'] = gdf['shortest']/gdf['length']

        # Calculate azimuth
        print("> Calculating azimuth")
        gdf['azimuth'] = [math.degrees(math.atan2((ln.xy[0][1] - ln.xy[0][0]), (ln.xy[1][1] - ln.xy[1][0]))) for ln in
                            gdf.geometry]
        return gdf

    def connectivity(self):
        gdf = self.gdf.copy()

        print("> Calculating connections")
        gdf['conn'] = p
        gdf['deadend'] = [1 if gdf.loc[i, 'conn'] < 2 else 0 for i in gdf.index]
        return gdf

    def visibility(self):
        gdf = self.gdf.copy()
        print("> Generating isovists")
        gdf['isovists'] = [Isovist(origin=geom.centroid, barriers=self.barriers).create() for geom in gdf['geometry']]
        gdf['iso_area'] = [geom.area for geom in gdf['isovists']]
        gdf['iso_perim'] = [geom.length for geom in gdf['isovists']]
        return gdf

    def greenery(self):
        gdf = self.gdf.copy()
        if self.trees is not None:
            trees = gpd.read_file(self.trees)
        else: print("Trees layer not found!")
        return gdf

    def all(self):
        self.gdf = self.dimension()
        self.gdf = self.direction()
        self.gdf = self.visibility()
        # self.gdf = self.connectivity()
        # self.gdf = self.greenery()
        return self.gdf

if __name__ == '__main__':
    directory = '/Volumes/Samsung_T5/Databases/CityOpenData'
    gbd = GeoBoundary('Vancouver, British Columbia')

    # Streets
    s = gpd.GeoDataFrame(Streets(
        gdf = gpd.read_file('https://opendata.vancouver.ca/explore/dataset/public-streets/download/?format=geojson&timezone=America/Los_Angeles&lang=en'),
        widths='https://opendata.vancouver.ca/explore/dataset/right-of-way-widths/download/?format=geojson&timezone=America/Los_Angeles&lang=en&epsg=26910',
        trees='https://opendata.vancouver.ca/explore/dataset/street-trees/download/?format=geojson&timezone=America/Los_Angeles&lang=en&epsg=26910'
    ).all())
    s = s.drop(
        ['from', 'to', 'name', 'lanes', 'service', 'ref', 'width_x', 'bridge', 'access', 'est_width', 'tunnel',
         'junction', 'index_right'], axis=1)
    s.to_file(f'{directory}/street-segments.shp', driver='ESRI Shapefile')

    # Buildings
    b = gpd.GeoDataFrame(Buildings(
        gdf=gpd.read_file(f'{directory}/building-footprints-2009.geojson'),
        to_crs=26910,
        group_by='bldgid',
        gb_func={
            'rooftype': 'max',
            'baseelev_m': 'min',
            'topelev_m': 'max',
            'maxht_m': 'max',
            'med_slope':'mean',
            'geometry': 'first'
        }
    ).all())
    b.to_file(f'{directory}/building-footprints.shp', driver='ESRI Shapefile')
