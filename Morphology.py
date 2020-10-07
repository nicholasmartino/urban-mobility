import geopandas as gpd
from shapely.geometry import Point
from shapely import ops


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
        self.gdf = self.triangulate()
        self.gdf = self.convex_hull()
        self.gdf = self.bounding_box()
        self.gdf = self.centroid()
        self.gdf = self.encl_circle()
        return self.gdf

class Streets:
    def __init__(self):
        return

if __name__ == '__main__':
    directory = '/Volumes/Samsung_T5/Databases/CityOpenData'
    b = Buildings(
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
    ).all()
    b = gpd.GeoDataFrame(b)
    b.to_file(f'{directory}/building-footprints-2009.shp', driver='ESRI Shapefile')
