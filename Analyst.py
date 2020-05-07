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

import glob
import timeit
import zipfile
from shutil import copyfile

import geopandas as gpd
import osmnx as ox
import pandana as pdna
import pandas as pd
import pylab as pl
import rasterio
import requests
import seaborn as sns
import statsmodels.api as sm
from PIL import Image
from Statistics.basic_stats import shannon_div
from graph_tool.all import *
from matplotlib.colors import ListedColormap
from pylab import *
from rasterio import features
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from shapely.affinity import translate, scale
from shapely.geometry import *
from shapely.ops import nearest_points
from skimage import morphology as mp


def download_file(url, filename=None):
    if filename is None: local_filename = url.split('/')[-1]
    else: local_filename = filename
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return local_filename


class GeoBoundary:
    def __init__(self, municipality='City, State', crs=26910,
                 directory='/Users/nicholasmartino/GoogleDrive/Geospatial/'):
        print(f"\nCreating GeoSpatial class {municipality}")
        self.municipality = str(municipality)
        self.directory = directory
        self.gpkg = f"{self.directory}Databases/{self.municipality}.gpkg"
        self.city_name = str(self.municipality).split(',')[0]
        self.crs = crs

        try:
            self.boundary = gpd.read_file(self.gpkg, layer='land_municipal_boundary')
            self.bbox = self.boundary.total_bounds
            print("> land_municipal_boundary layer found")
        except: print("> land_municipal_boundary layer not found")

        try:
            self.nodes = gpd.read_file(self.gpkg, layer='network_nodes')
            self.links = gpd.read_file(self.gpkg, layer='network_links')
            print("> network_nodes & _links layers found")
        except: print("> network_nodes &| _links layer(s) not found")

        # try:
        #     self.DAs = gpd.read_file(self.gpkg, layer='land_dissemination_area')
        #     print("> land_dissemination_area layer found")
        # except: print('> land_dissemination_area layer not found')

        # try:
        #     self.properties = gpd.read_file(self.gpkg, layer='land_assessment_fabric')
        #     self.parcels = gpd.read_file(self.gpkg, layer='land_assessment_parcels')
        #     print("> land_assessment_fabric and land_assessment_parcels layers found")
        # except: print("> land_assessment_fabric &| land_assessment_parcels layer(s) not found")

        # try:
        #     self.walking_net = gpd.read_file(self.gpkg, layer='network_links_walking')
        #     self.cycling_net = gpd.read_file(self.gpkg, layer='network_links_cycling')
        #     self.driving_net = gpd.read_file(self.gpkg, layer='network_links_driving')
        #     print('> _walking, _cycling and _driving network_links layers found')
        # except: print('> network_links_{walking &| cycling &| driving} layer(s) not found')

        # try:
        #     self.cycling = gpd.read_file(self.gpkg, layer='network_cycling_official')
        #     print('> network_cycling_official layer found')
        # except: print('> network_cycling_official layer not found')

        print(f"Class {self.city_name} created @ {datetime.datetime.now()}, crs {self.crs}\n")

    # Download and pre process data
    def update_databases(self, bound=True, net=True, census=False, icbc=False):
        # Download administrative boundary from OpenStreetMaps
        if bound:
            print(f"Downloading {self.city_name}'s administrative boundary from OpenStreetMaps")
            self.boundary = ox.gdf_from_place(self.municipality)
            # self.boundary.to_crs(crs=4326, epsg=4326, inplace=True)
            # self.boundary.to_crs(crs=26910, epsg=26910, inplace=True)
            self.boundary.to_file(self.gpkg, layer='land_municipal_boundary', driver='GPKG')
            self.boundary = gpd.read_file(self.gpkg, layer='land_municipal_boundary')
            self.boundary.to_crs(epsg=self.crs, inplace=True)
            self.bbox = self.boundary.total_bounds
            s_index = self.boundary.sindex

        # Download street networks from OpenStreetMaps
        if net:
            print(f"Downloading {self.city_name}'s street network from OpenStreetMaps")
            network = ox.graph_from_place(self.municipality)
            ox.save_graph_shapefile(network, 'osm', self.directory)
            edges = gpd.read_file(self.directory+'osm/edges/edges.shp')
            nodes = gpd.read_file(self.directory+'osm/nodes/nodes.shp')
            edges.crs = 4326  # {'init': 'epsg:4326'}
            edges.to_crs(epsg=self.crs, inplace=True)
            edges.to_file(self.gpkg, layer='network_links', driver='GPKG')
            nodes.crs = 4326  # {'init': 'epsg:4326'}
            nodes.to_crs(epsg=self.crs, inplace=True)
            nodes.to_file(self.gpkg, layer='network_nodes', driver='GPKG')

            # Simplify links
            s_tol = 15
            s_links = edges
            s_links.geometry = edges.simplify(s_tol)

            sbb = False
            if sbb:
                # Buffer endpoints and get centroid
                end_pts = gpd.GeoDataFrame(geometry=[Point(l.xy[0][0], l.xy[1][0]) for l in s_links.geometry] +
                                                    [Point(l.xy[0][1], l.xy[1][1]) for l in s_links.geometry])
                uu = gpd.GeoDataFrame(geometry=[pol.centroid for pol in end_pts.buffer(s_tol/2).unary_union]).unary_union

                # Snap edges to vertices
                lns = []
                for ln in s_links.geometry:
                    p0 = Point(ln.coords[0])
                    p1 = Point(ln.coords[1])
                    np0 = nearest_points(p0, uu)[1]
                    np1 = nearest_points(p1, uu)[1]
                    lns.append(LineString([np0, np1]))

                s_links['geometry'] = lns

            s_links.to_file(self.gpkg, layer='network_links_simplified')
            print("Street network from OpenStreetMap updated")

    def merge_csv(self, path):
        os.chdir(path)
        file_out = "merged.csv"
        if os.path.exists(file_out):
            os.remove(file_out)
        file_pattern = ".csv"
        list_of_files = [file for file in glob.glob('*'+file_pattern)]
        print(list_of_files)
        # Consolidate all CSV files into one object
        result_obj = pd.concat([pd.read_csv(file) for file in list_of_files])
        # Convert the above object into a csv file and export
        result_obj.to_csv(file_out, index=False, encoding="utf-8")
        df = pd.read_csv("merged.csv")
        full_path = os.path.realpath(__file__)
        path, filename = os.path.split(full_path)
        os.chdir(path)
        print('CSVs successfully merged')
        return df

    def elevation(self, hgt_file, lon, lat):
        SAMPLES = 1201  # Change this to 3601 for SRTM1
        with open(hgt_file, 'rb') as hgt_data:
            # Each data is 16bit signed integer(i2) - big endian(>)
            elevations = np.fromfile(hgt_data, np.dtype('>i2'), SAMPLES * SAMPLES) \
                .reshape((SAMPLES, SAMPLES))

            lat_row = int(round((lat - int(lat)) * (SAMPLES - 1), 0))
            lon_row = int(round((lon - int(lon)) * (SAMPLES - 1), 0))

            return elevations[SAMPLES - 1 - lat_row, lon_row].astype(int)

    # Network analysis
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

    def centrality(self, run=True, dual=False):
        if run:
            layer = 'network_links'
            links = gpd.read_file(self.gpkg, layer=layer)
            nodes = gpd.read_file(self.gpkg, layer='network_nodes')
            start_time = timeit.default_timer()

            # Calculate azimuth of links
            links['azimuth'] = [math.degrees(math.atan2((ln.xy[0][1] - ln.xy[0][0]), (ln.xy[1][1] - ln.xy[1][0]))) for
                                ln in links.geometry]
            pos_000_090 = links['azimuth'].loc[(links['azimuth'] > 0) & (links['azimuth'] < 90)]
            pos_090_180 = links['azimuth'].loc[(links['azimuth'] > 90) & (links['azimuth'] < 180)]
            neg_000_090 = links['azimuth'].loc[(links['azimuth'] < 0) & (links['azimuth'] > -90)]
            neg_090_180 = links['azimuth'].loc[(links['azimuth'] < -90)]
            tdf = pd.concat([pos_000_090, (90 - (pos_090_180 - 90)), neg_000_090 * -1, neg_090_180 + 180])
            links['azimuth_n'] = tdf
            tdf = None

            """
            # Extract nodes from links
            links.dropna(subset=['geometry'], inplace=True)
            links.reset_index(inplace=True, drop=True)
            print(f"Processing centrality measures for {len(links)} links")
            l_nodes = gpd.GeoDataFrame(geometry=[Point(l.xy[0][0], l.xy[1][0]) for l in links.geometry]+
                                                [Point(l.xy[0][1], l.xy[1][1]) for l in links.geometry])

            # Pre process nodes
            rf = 3
            l_nodes['cid'] = [f'%.{rf}f_' % n.xy[0][0] + f'%.{rf}f' % n.xy[1][0] for n in l_nodes.geometry]
            l_nodes.drop_duplicates('cid', inplace=True)
            l_nodes.reset_index(inplace=True, drop=True)

            # Create location based id
            links['o_cid'] = [f'%.{rf}f_' % l.xy[0][0] + f'%.{rf}f' % l.xy[1][0] for l in links.geometry]
            links['d_cid'] = [f'%.{rf}f_' % l.xy[0][1] + f'%.{rf}f' % l.xy[1][1] for l in links.geometry]
            """

            # Create topological graph and add vertices
            osm_g = Graph(directed=False)

            for i in list(nodes.index):
                v = osm_g.add_vertex()
                v.index = int(i)

            print(f"> {len(list(osm_g.vertices()))} vertices added to graph, {len(nodes)} nodes downloaded from OSM")

            # Graph from OSM topological data
            links_ids = []
            osm_g_edges = []
            for i in list(links.index):
                o_osmid = links.at[i, 'from']
                d_osmid = links.at[i, 'to']
                o_id = nodes.loc[(nodes['osmid'] == o_osmid)].index[0]
                d_id = nodes.loc[(nodes['osmid'] == d_osmid)].index[0]
                if o_osmid != d_osmid:
                    land_length = links.at[i, 'geometry'].length
                    topo_length = LineString([
                        nodes.loc[nodes.osmid == o_osmid].geometry.values[0],
                        nodes.loc[nodes.osmid == d_osmid].geometry.values[0]
                    ]).length
                    w = int(((land_length / topo_length) * land_length)/100)
                    osm_g_edges.append([int(o_id), int(d_id), w])
                    links_ids.append(i)

            straightness = osm_g.new_edge_property("int16_t")
            edge_properties = [straightness]
            print("> All links and weights listed, creating graph")
            osm_g.add_edge_list(osm_g_edges, eprops=edge_properties)

            """
            # Add edges
            g_edges = []
            for i, l in enumerate(links.geometry):
                o = l_nodes.loc[l_nodes['cid'] == f'%.{rf}f_' % l.xy[0][0] + f'%.{rf}f' % l.xy[1][0]].index[0]
                d = l_nodes.loc[l_nodes['cid'] == f'%.{rf}f_' % l.xy[0][1] + f'%.{rf}f' % l.xy[1][1]].index[0]
                g_edges.append([o, d, links.at[i, 'azimuth_n']])

            g_azimuth = g.new_edge_property("double")
            edge_properties = [g_azimuth]
            g.add_edge_list(g_edges, eprops=edge_properties)

            # Create topological dual graph
            dg = Graph(directed=False)

            # Iterate over network links indexes to create nodes of dual graph
            for i in list(links.index):
                v = dg.add_vertex()
                v.index = i

            # Iterate over network links geometries to create edges of dual graph
            azs = []
            dg_edges = []
            for l, i in zip(links.geometry, list(links.index)):
                # Get other links connected to this link
                o = links.loc[links['o_cid'] == f'%.{rf}f_' % l.xy[0][0] + f'%.{rf}f' % l.xy[1][0]]
                d = links.loc[links['d_cid'] == f'%.{rf}f_' % l.xy[0][1] + f'%.{rf}f' % l.xy[1][1]]

                conn = pd.concat([o, d])
                conn.drop_duplicates(inplace=True)

                # List edges and azimuths for dual graph
                azs1 = []
                for j in list(conn.index):
                    azimuths = [links.at[i, 'azimuth_n'], links.at[j, 'azimuth_n']]
                    w = 1/(max(azimuths) - min(azimuths))
                    dg_edges.append([i, j, w])
                    azs1.append(w)
                azs.append(sum(azs1)/len(azs1))

            # Add edges to dual graph
            dg_azimuth = dg.new_edge_property("double")
            eprops = [dg_azimuth]
            dg.add_edge_list(dg_edges, eprops=eprops)

            if dual: g = dg
            
            links['n_choice'] = np.log(links['choice'])
            links['choice'] = betweenness(dg)[0].get_array()
            links['choice'].replace([np.inf, -np.inf], 0, inplace=True)
            links['n_choice'].replace([np.inf, -np.inf], 0, inplace=True)
            """

            # Calculate centrality measures and assign to nodes
            nodes['closeness'] = closeness(osm_g).get_array()
            btw = betweenness(osm_g)
            nodes['betweenness'] = btw[0].get_array()

            # Assign betweenness to links
            l_betweenness = pd.DataFrame(links_ids, columns=['ids'])
            l_betweenness['betweenness'] = btw[1].get_array()
            l_betweenness.index = l_betweenness.ids
            for i in links_ids:
                links.at[i, 'betweenness'] = l_betweenness.at[i, 'betweenness']

            # Normalize betweenness
            nodes['n_betweenness'] = np.log(nodes['betweenness'])
            links['n_betweenness'] = np.log(links['betweenness'])

            # Replace infinity and NaN values
            rep = lambda col: col.replace([np.inf, -np.inf], np.nan, inplace=True)
            for c in [nodes['closeness'], nodes['betweenness'], nodes['n_betweenness'],
                      links['betweenness'], links['n_betweenness']]:
                rep(c)

            # Export to GeoPackage
            links.to_file(self.gpkg, layer=layer)
            nodes.to_file(self.gpkg, layer='network_nodes')

            elapsed = round((timeit.default_timer() - start_time) / 60, 1)
            print(f"Centrality measures processed in {elapsed} minutes")
            return links

    def filter_networks(self):
        # Filter Open Street Map Networks into Walking, Cycling and Driving
        walking = ['bridleway', 'corridor', 'footway', 'living_street', 'path', 'pedestrian', 'residential',
                   'road', 'secondary', 'service', 'steps', 'tertiary', 'track', 'trunk']
        self.walking_net = self.links.loc[self.links.highway.apply(lambda x: any(element in x for element in walking))]

        cycling = ['bridleway', 'corridor', 'cycleway', 'footway', 'living_street', 'path', 'pedestrian',
                   'residential', 'road', 'secondary', 'service', 'tertiary', 'track', 'trunk']
        self.cycling_net = self.links.loc[self.links.highway.apply(lambda x: any(element in x for element in cycling))]

        driving = ['corridor', 'living_street', 'motorway', 'primary', 'primary_link', 'residential', 'road',
                   'secondary', 'secondary_link', 'service', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link',
                   'unclassified']
        self.driving_net = self.links.loc[self.links.highway.apply(lambda x: any(element in x for element in driving))]

        self.walking_net.to_file(self.gpkg, layer='network_links_walking')
        self.cycling_net.to_file(self.gpkg, layer='network_links_cycling')
        self.driving_net.to_file(self.gpkg, layer='network_links_driving')
        return None

    def network_analysis(self, sample_gdf, aggregated_layers, service_areas):
        """
        Given a layer of spatial features, it aggregates data from its surroundings using network service areas

        :param sample_layer: (str) Sample features to be analyzed, ex: 'lda' or 'parcel'.
        :param aggregated_layers: (dict) Layers and columns to aggregate data, ex: {'lda':["walk"], 'parcel':["area"]}
        :param service_areas: (list) Buffer to aggregate from each sample_layer feature[400, 800, 1600]
        :return:
        """

        print(f'> Network analysis for {len(sample_gdf.geometry)} geometries at {service_areas} buffer radius')
        start_time = timeit.default_timer()

        # Load data
        nodes = self.nodes
        edges = self.links
        print(nodes.head(3))
        print(edges.head(3))
        nodes.index = nodes.osmid

        # Reproject GeoDataFrames
        sample_gdf.to_crs(epsg=self.crs, inplace=True)
        nodes.to_crs(epsg=self.crs, inplace=True)
        edges.to_crs(epsg=self.crs, inplace=True)

        # Create network
        net = pdna.Network(nodes.geometry.x,
                           nodes.geometry.y,
                           edges["from"],
                           edges["to"],
                           edges[["length"]],
                           twoway=True)
        print(net)
        net.precompute(max(service_areas))

        x, y = sample_gdf.centroid.x, sample_gdf.centroid.y
        sample_gdf["node_ids"] = net.get_node_ids(x.values, y.values)

        buffers = {}
        for key, values in aggregated_layers.items():
            gdf = gpd.read_file(self.gpkg, layer=key)
            gdf.to_crs(epsg=self.crs, inplace=True)
            x, y = gdf.centroid.x, gdf.centroid.y
            gdf["node_ids"] = net.get_node_ids(x.values, y.values)
            gdf[f"{key}_ct"] = 1

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
                print(f'Processing {value} column from {key} GeoDataFrame')
                net.set(node_ids=gdf["node_ids"], variable=gdf[value])

                for radius in service_areas:

                    cnt = net.aggregate(distance=radius, type="count", decay="flat")
                    sum = net.aggregate(distance=radius, type="sum", decay="flat")
                    ave = net.aggregate(distance=radius, type="ave", decay='flat')

                    sample_gdf[f"{value}_r{radius}_cnt"] = list(cnt.loc[sample_gdf["node_ids"]])
                    sample_gdf[f"{value}_r{radius}_sum"] = list(sum.loc[sample_gdf["node_ids"]])
                    sample_gdf[f"{value}_r{radius}_ave"] = list(ave.loc[sample_gdf["node_ids"]])

                    sample_gdf.to_file(self.gpkg, layer=f'network_analysis', driver='GPKG')

        elapsed = round((timeit.default_timer() - start_time) / 60, 1)
        sample_gdf.to_file(self.gpkg, layer=f'network_analysis', driver='GPKG')
        print(f'Network analysis processed in {elapsed} minutes @ {datetime.datetime.now()}')

    def network_from_polygons(self, filepath='.gpkg', layer='land_assessment_parcels', remove_islands=False,
                              scale_factor=0.82, tolerance=4, buffer_radius=10, min_lsize=20, max_linters=0.5):
        """
        Input a set of polygons and generate linear networks within the center of empty spaces among features.

        Params:
        filepath (str) = Directory for the GeoDatabase (i.e.: .gdb, .gpkg) with the polygons
        layer (str) = Polygon layer name within the GeoDatabase
        tolerance (float) = Tolerance for edges simplification
        buffer_radius (float) = Radius of intersection buffer for node simplification
        """
        s = 0
        figname = 'hq'
        sf = scale_factor

        print(f"> Processing centerlines for {layer} from {self.gpkg}")
        start_time = timeit.default_timer()

        # Read GeoDatabase
        gdf = gpd.read_file(filepath, layer=layer, driver='GPKG')
        gdf.dropna(subset=['geometry'], inplace=True)
        gdf.to_crs(epsg=self.crs, inplace=True)
        gdf_uu = gdf.geometry.unary_union

        # Extract open spaces
        try: chull = gpd.GeoDataFrame(geometry=[self.boundary.buffer(10)], crs=gdf.crs)
        except: chull = gpd.GeoDataFrame(geometry=[gdf_uu.convex_hull.buffer(10)], crs=gdf.crs)
        empty = gpd.overlay(chull, gdf, how='difference')

        # Export open spaces to image file
        empty.plot()
        plt.axis('off')
        plt.savefig(f'{figname}.png', dpi=600)

        # Create network_from_polygons from black and white raster
        tun = 1 - pl.imread(f'{figname}.png')[..., 0]
        skl = mp.medial_axis(tun)

        # Display and save centerlines
        image = Image.fromarray(skl)
        image.save(f'{figname}_skltn.png')

        # Load centerlines image
        with rasterio.open(f'{figname}_skltn.png') as src:
            blue = src.read()
        mask = blue != 0

        # Transform raster into shapely geometry (vectorize)
        shapes = features.shapes(blue, mask=mask)
        cl_pxl = gpd.GeoDataFrame(geometry=[Polygon(shape[0]['coordinates'][0]) for shape in shapes], crs=gdf.crs)

        # Buffer polygons to form centerline polygon
        cl_pxl_sc = gpd.GeoDataFrame(geometry=[scale(cl_pxl.buffer(-0.1).unary_union, sf, -sf, sf)], crs=gdf.crs)

        # Geo reference edges based on centroids
        raw_centr = gdf.unary_union.convex_hull.buffer(10).centroid
        xoff = raw_centr.x - cl_pxl_sc.unary_union.convex_hull.centroid.x  # dela.centroid.x
        yoff = raw_centr.y - cl_pxl_sc.unary_union.convex_hull.centroid.y  # dela.centroid.y

        # Translate, scale down and export
        cl_pxl_tr = gpd.GeoDataFrame(geometry=[translate(cl_pxl_sc.unary_union, xoff=xoff, yoff=yoff, zoff=0.0)], crs=gdf.crs)

        # Intersect pixels and vectorized center line to identify potential nodes of the network
        cl_b_mpol = gpd.GeoDataFrame(geometry=[cl_pxl_tr.buffer(2).unary_union], crs=gdf.crs)

        # Negative buffer to find potential nodes
        buffer_r = -2.8
        print(f"> {len(cl_b_mpol.buffer(buffer_r).geometry[0])} potential nodes identified")

        # Buffer and subtract
        node_buffer = gpd.GeoDataFrame(
            geometry=[pol.centroid.buffer(buffer_radius) for pol in cl_b_mpol.buffer(buffer_r).geometry[0]],
            crs=gdf.crs)
        difference = gpd.overlay(node_buffer, cl_b_mpol, how="difference")
        difference['mpol_len'] = [len(mpol) if type(mpol)==type(MultiPolygon()) else 1 for mpol in difference.geometry]
        p_nodes = difference.loc[difference['mpol_len'] > 2]

        # Extract nodes that intersect more than two links
        node = node_buffer.iloc[difference.index]
        node['n_links'] = difference['mpol_len']
        node = node.loc[node['n_links'] > 2].centroid
        node = gpd.GeoDataFrame(geometry=[pol.centroid for pol in node.buffer(6).unary_union], crs=gdf.crs)

        # Buffer extracted nodes
        cl_b2 = gpd.GeoDataFrame(geometry=cl_pxl_tr.buffer(2).boundary)
        cl_b1 = gpd.GeoDataFrame(geometry=cl_pxl_tr.buffer(1))
        cl_b1.to_file(self.gpkg, layer=f'network_centerline')

        # Subtract buffered nodes from center line polygon
        node_b6 = gpd.GeoDataFrame(geometry=node.buffer(6), crs=gdf.crs)
        node_b9 = gpd.GeoDataFrame(geometry=node.buffer(9), crs=gdf.crs)
        node_b12 = gpd.GeoDataFrame(geometry=node.buffer(12), crs=gdf.crs)

        # Subtract buffered nodes from centerline polygon and simplify
        links = gpd.overlay(cl_b2, node_b6, how='difference').simplify(tolerance)

        # Find link vertices (changes in direction)
        snapping = gpd.GeoDataFrame()
        for ln in links.geometry[0]:
            # Extract vertices from lines and collapse close vertices
            vertices = gpd.GeoDataFrame(geometry=[Point(coord) for coord in ln.coords], crs=gdf.crs)
            try: vertices = gpd.GeoDataFrame(geometry=[pol.centroid for pol in vertices.buffer(buffer_radius).unary_union], crs=gdf.crs)
            except: vertices = gpd.GeoDataFrame(geometry=vertices.buffer(buffer_radius).centroid, crs=gdf.crs)
            # Eliminate vertices if its buffer intersects with the network_nodes
            vertices = vertices[vertices.disjoint(node_b6.unary_union)]
            snapping = pd.concat([snapping, vertices])
        # Simplify and export
        snapping.reset_index(inplace=True)
        vertices = gpd.GeoDataFrame(geometry=[pol.centroid for pol in snapping.buffer(buffer_radius).unary_union], crs=gdf.crs)
        vertices = vertices[vertices.disjoint(node_b12.unary_union)]
        vertices = pd.concat([vertices, node])

        # Extract and explode line segments
        links_exploded = []
        for ln in links.geometry[0]:
            if type(ln) == type(MultiLineString()):
                coords = [l.coords for l in ln]
            else: coords = ln.coords
            for i, coord in enumerate(coords):
                if i < len(coords)-1: links_exploded.append(LineString([Point(coords[i]), Point(coords[i+1])]))
        links_e = gpd.GeoDataFrame(geometry=links_exploded, crs=gdf.crs)

        # Snap edges to vertices
        lns = []
        for ln in links_exploded:
            p0 = Point(ln.coords[0])
            p1 = Point(ln.coords[1])
            np0 = nearest_points(p0, vertices.unary_union)[1]
            np1 = nearest_points(p1, vertices.unary_union)[1]
            lns.append(LineString([np0, np1]))

        # Create GeoPackage with links
        edges = gpd.GeoDataFrame(geometry=lns, crs=gdf.crs)

        # Drop links smaller than a certain length only connected to one node
        for i, link in enumerate(edges.geometry):
            if float(link.length) < min_lsize:
                try: len(link.intersection(node_b6.unary_union))
                except:
                    edges.drop(index=i, inplace=True)
                    print(f"> Link at index {i} have only one connected node and its length is smaller than threshold")

        # Create centroid id field, drop duplicate geometry and export to GeoPackage
        edges['cid'] = [str(ln.centroid) for ln in edges.geometry]
        edges.drop_duplicates(['cid'], inplace=True)
        edges.reset_index(inplace=True, drop=True)
        edges['index'] = list(edges.index)
        edges['azimuth'] = [math.degrees(math.atan2((ln.xy[0][1] - ln.xy[0][0]), (ln.xy[1][1] - ln.xy[1][0]))) for ln in
                            edges.geometry]
        edges['length'] = [ln.length for ln in edges.geometry]

        vertices.reset_index(inplace=True)
        # Iterate over vertices
        for i, v in enumerate(vertices.geometry):
            # Remove isolated vertices
            if v.buffer(2).disjoint(edges.unary_union):
                vertices.drop(index=i, inplace=True)

            # Remove lines with close azimuth
            edges_in = edges[edges.intersects(v)]
            edges_in.reset_index(inplace=True)

            if len(edges_in) == 1: pass
            else:
                # Compare origin, destination and centroids of each line intersecting vertices
                for i, ln0 in enumerate(edges_in.geometry):
                    # If iteration is in the last item set ln1 to be the first line
                    if i == len(edges_in)-1:
                        li1 = edges_in.at[0, 'index']
                        ln1 = edges_in.at[0, 'geometry']
                    else:
                        li1 = edges_in.at[i+1, 'index']
                        ln1 = edges_in.at[i+1, 'geometry']

                    inters_bf = 4
                    inters0 = ln0.buffer(inters_bf).intersection(ln1.buffer(inters_bf)).area/ln0.buffer(inters_bf).area
                    inters1 = ln1.buffer(inters_bf).intersection(ln0.buffer(inters_bf)).area/ln1.buffer(inters_bf).area
                    inters = max(inters0, inters1)

                    li0 = edges_in.at[i, 'index']
                    if inters > max_linters:
                        if ln0.length < ln1.length:
                            try: edges.drop(li0, axis=0, inplace=True)
                            except: pass
                            print(f"> Link {li0} dropped due to similarity with another edge above threshold {max_linters}")
                        else:
                            try: edges.drop(li1, axis=0, inplace=True)
                            except: pass
                            print(f"> Link {li1} dropped due to similarity with another edge above threshold {max_linters}")

        # Remove nodes that are not intersections
        edges_b2 = gpd.GeoDataFrame(geometry=[edges.buffer(2).unary_union])
        difference = gpd.overlay(node_b6, edges_b2, how="difference")
        difference['mpol_len'] = [len(mpol) if type(mpol)==type(MultiPolygon()) else 1 for mpol in difference.geometry]
        node = node.loc[difference['mpol_len'] > 2]
        node.to_file(self.gpkg, layer='network_nodes')

        # Remove islands
        if remove_islands: edges = edges[edges.intersects(node.unary_union)]

        # Export links and vertices
        edges.to_file(self.gpkg, driver='GPKG', layer=f'network_links')
        vertices.to_file(self.gpkg, layer='network_vertices')

        elapsed = round((timeit.default_timer() - start_time) / 60, 1)
        return print(f"Centerlines processed in {elapsed} minutes @ {datetime.datetime.now()}")

    # Spatial analysis
    def set_parameters(self, service_areas, unit='lda', samples=None, max_area=7000000, elab_name='Sunset', bckp=True,
                       layer='Optional GeoPackage layer to analyze', buffer_type='circular'):
        # Load GeoDataFrame and assign layer name for LDA
        if unit == 'lda':
            gdf = self.DAs.loc[self.DAs.geometry.area < max_area]
            layer = 'land_dissemination_area'

        # Pre process database for elementslab 1600x1600m 'Sandbox'
        elif unit == 'elab_sandbox':
            self.directory = 'Sandbox/'+elab_name
            self.gpkg = elab_name+'.gpkg'
            if 'PRCLS' in layer:
                nodes_gdf = gpd.read_file(self.gpkg, layer='network_intersections')
                links_gdf = gpd.read_file(self.gpkg, layer='network_streets')
                cycling_gdf = gpd.read_file(self.gpkg, layer='network_cycling')
                if '2020' in layer:
                    self.nodes = nodes_gdf.loc[nodes_gdf['ctrld2020'] == 1]
                    self.links = links_gdf.loc[links_gdf['new'] == 0]
                    self.cycling = cycling_gdf.loc[cycling_gdf['year'] == '2020-01-01']
                    self.cycling['type'] = self.cycling['type2020']
                    self.cycling.reset_index(inplace=True)
                elif '2050' in layer:
                    self.nodes = nodes_gdf.loc[nodes_gdf['ctrld2050'] == 1]
                    self.links = links_gdf
                    self.cycling = cycling_gdf
                    self.cycling['type'] = cycling_gdf['type2050']
            self.properties = gpd.read_file(self.gpkg, layer=layer)
            self.properties.crs = {'init': 'epsg:26910'}

            # Reclassify land uses and create bedroom and bathroom columns
            uses = {'residential': ['RS_SF_D', 'RS_SF_A', 'RS_MF_L', 'RS_MF_H'],
                    'retail': ['CM', 'MX'],
                    'civic': ['CV'],
                    'green': ['OS']}
            new_uses = []
            index = list(self.properties.columns).index("LANDUSE")
            all_prim_uses = [item for sublist in list(uses.values()) for item in sublist]
            for row in self.properties.iterrows():
                for key, value in uses.items():
                    if row[1]['LANDUSE'] in value:
                        new_uses.append(key)
                if row[1]['LANDUSE'] not in all_prim_uses:
                    new_uses.append(row[1]['LANDUSE'])
            self.properties['elab_use'] = new_uses
            self.properties['PRIMARY_ACTUAL_USE'] = self.properties['LANDUSE']
            self.properties['NUMBER_OF_BEDROOMS'] = 2
            self.properties['NUMBER_OF_BATHROOMS'] = 1

            # Define GeoDataFrame
            # gdf = gpd.GeoDataFrame(geometry=self.properties.unary_union.convex_hull)
            gdf = self.properties[['OBJECTID', 'geometry']]
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
        if 'index_right' in self.properties.columns:
            self.properties.drop('index_right', axis=1, inplace=True)
        if 'index_left' in self.properties.columns:
            self.properties.drop('index_left', axis=1, inplace=True)

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

            jgdf = gpd.sjoin(geom, self.properties, how='right', op="intersects")
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
        if 'index_right' in self.properties.columns:
            self.properties.drop('index_right', axis=1, inplace=True)
        if 'index_left' in self.properties.columns:
            self.properties.drop('index_left', axis=1, inplace=True)

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

            jgdf = gpd.sjoin(geom, self.properties, how='right', op="intersects")
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

    def street_network_indicators(self, net_simperance=10):
        # Define GeoDataframe sample_layer unit
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
                    nodes_w = nodes_w.geometry.buffer(net_simperance).unary_union
                    len_nodes_w = len(nodes_w)
                    if len(nodes_w) == 0:
                        len_nodes_w = 1
                except:
                    exceptions.append('exception')
                    len_nodes_w = 1
                intrs_den[key].append(round(len_nodes_w / (pol.area / 10000), 2))
                edges_w = self.links[self.links.geometry.within(pol)]
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

    # Process results
    def regression(self):
        """
        Reference: https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf
        """

        gdf = self.params['gdf']
        sas = self.params['service_areas']

        pg_gdf = gpd.read_file('/Users/nicholasmartino/GoogleDrive/Geospatial/Databases/Prince George, British Columbia.gpkg', layer='land_dissemination_area')
        van_gdf = gpd.read_file('/Users/nicholasmartino/GoogleDrive/Geospatial/Databases/Metro Vancouver, British Columbia.gpkg', layer='land_dissemination_area')
        gdf = pd.concat([pg_gdf, van_gdf])

        # Get name of urban form features analyzed within the service areas
        x_features = []
        for col in gdf.columns:
            for radius in sas:
                id = f'_r{radius}m'
                if id in col[len(id):]:
                    x_features.append(col)

        # Get y-variables
        gdf['drive'] = gdf['car_driver']+gdf['car_passenger']
        y_features = ['walk', 'bike', 'drive', 'bus']
        y_gdf = gdf[y_features]
        y_gdf.dropna(inplace=True)

        # Calculate correlation among features
        x_gdf = gdf[x_features]
        x_gdf.dropna(inplace=True, axis=1)
        corr = x_gdf.corr()
        sns.heatmap(corr)
        plt.show()

        # Drop correlations higher than 90%
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= 0.9:
                    if columns[j]:
                        columns[j] = False
        selected_columns = x_gdf.columns[columns]
        x_gdf = x_gdf[selected_columns]
        x_gdf = x_gdf.iloc[y_gdf.index]

        # Calculate p-values
        x = x_gdf.values
        y = y_gdf.values
        p = pd.DataFrame()
        p['feature'] = selected_columns

        for i, col in enumerate(y_gdf.columns):
            regressor_ols = sm.OLS(y.transpose()[i-1], x).fit()
            with open(f'Regression/{datetime.datetime.now()}_{col}.txt', 'w') as file:
                file.write(str(regressor_ols.summary()))
            p[f'{col}_pv'] = regressor_ols.pvalues

        # Select n highest p-values for each Y
        highest = pd.DataFrame()
        for i, col in enumerate(y_gdf.columns):
            srtd = p.sort_values(by=f'{col}_pv', ascending=False)
            highest[f'{col}'] = list(srtd.head(3)['feature'])

        return

    def network_report(self):
        nodes_gdf = gpd.read_file(self.gpkg, layer='network_nodes')
        links_gdf = gpd.read_file(self.gpkg, layer='network_links')

        # Setup directory parameters
        save_dir = f"{self.directory}Reports/"
        if 'Reports' in os.listdir(self.directory): pass
        else: os.mkdir(save_dir)
        if self.municipality in os.listdir(save_dir): pass
        else: os.mkdir(f"{save_dir}{self.municipality}")

        # Calculate boundary area
        df = pd.DataFrame()
        try:
            self.boundary = self.boundary.to_crs(3157)
            bounds = self.boundary.area[0]/10000
        except:
            print(f'No boundary found, using convex hull')
            nodes_gdf.crs = 3157
            links_gdf.crs = 3157
            bounds = links_gdf.unary_union.convex_hull.area/10000
        print(f'Area: {bounds} ha')

        links_gdf_bf = gpd.GeoDataFrame(geometry=[links_gdf.buffer(1).unary_union])
        nodes_gdf_bf = gpd.GeoDataFrame(geometry=[nodes_gdf.buffer(7).unary_union])
        links_gdf_sf = gpd.GeoDataFrame(geometry=[l for l in gpd.overlay(links_gdf_bf, nodes_gdf_bf, how='difference').geometry[0]])

        # Calculate basic network indicators
        print(f"> Calculating basic network stats")
        df['Area'] = [format(bounds, '.2f')]
        df['Node count'] = [format(len(nodes_gdf), '.2f')]
        df['Link count'] = [format(len(links_gdf_sf), '.2f')]
        df['Node Density (nodes/ha)'] = [format(len(nodes_gdf)/bounds, '.2f')]
        df['Link Density (links/ha)'] = [format(len(links_gdf_sf)/bounds, '.2f')]
        df['Link-Node Ratio (count)'] = [format(len(links_gdf_sf)/len(nodes_gdf), '.2f')]
        df['Average Link Length (meters)'] = [format(sum([(ln.area) for ln in links_gdf_sf.geometry])/len(links_gdf_sf), '.2f')]
        df = df.transpose()
        df.index.name = 'Indicator'
        df.columns = ['Measure']

        # Define image properties
        fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [6, 1]})
        fig.set_size_inches(7.5, 7.5)
        ax0.axis('off')
        ax1.axis('off')

        # Plot map and table
        ax0.set_title(f'Network Indicators - {self.municipality}')
        links_gdf.buffer(4).plot(ax=ax0, facecolor='black', linewidth=0.5, linestyle='solid')
        nodes_gdf.buffer(8).plot(ax=ax0, edgecolor='black', facecolor='white', linewidth=0.5, linestyle='solid')
        ax1.table(
            cellText=df.values,
            colLabels=df.columns,
            colWidths=[0.1],
            rowLabels=df.index,
            loc='right',
            edges='horizontal')

        # Setup and save figure
        plt.savefig(f"{save_dir}{self.municipality}.png", dpi=300)

        # Plot centrality measures if exists
        if 'betweenness' in links_gdf.columns:
            links_gdf.plot(column='betweenness', cmap='viridis_r', legend=True)
            fig.set_size_inches(7.5, 7.5)
            plt.axis('off')
            plt.title(f'Betweenness - {self.municipality}')
            plt.savefig(f"{save_dir}{self.municipality}_bt.png", dpi=300)

        df['Measure'] = pd.to_numeric(df['Measure'])
        print(f"Report successfully saved at {self.directory}")
        return df

    def export_map(self):
        
        # Process geometry
        boundaries = self.DAs.geometry.boundary
        centroids = gpd.GeoDataFrame(geometry=self.DAs.geometry.centroid)
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
        dest_gdf = self.properties.loc[(self.properties['elab_use'] == 'retail') |
                                    (self.properties['elab_use'] == 'office') |
                                    (self.properties['elab_use'] == 'entertainment')]
        dest_gdf['geometry'] = dest_gdf.geometry.centroid
        dest_gdf.drop_duplicates('geometry')
        dest_gdf.to_file(self.directory+'Shapefiles/'+self.municipality+' - Destinations.shp', driver='ESRI Shapefile')
        return self

    def export_parcels(self):
        gdf = self.properties
        gdf.to_file('Shapefiles/' + self.params['layer'], driver='ESRI Shapefile')
        for col in gdf.columns:
            if str(type(gdf[col][0])) == "<class 'numpy.float64'>" or str(type(gdf[col][0])) == "<class 'numpy.int64'>" or col == "LANDUSE":
                if sum(gdf[col]) == 0:
                    gdf.drop(col, inplace=True, axis=1)
                    print(col + ' column removed')
        gdf.to_file('Shapefiles/'+self.params['layer']+'_num', driver='ESRI Shapefile')
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


if __name__ == '__main__':
    BUILD_REAL_NETWORK = False
    BUILD_PROXY_NETWORK = False
    NETWORK_STATS = False
    VERSIONS = ["", '2']

    osm = GeoBoundary(municipality='Hillside Quadra, Victoria, British Columbia', crs=26910)
    osm.update_databases(bound=True)
    osm.centrality()

    real_auto = GeoBoundary(municipality=f'Hillside Quadra', crs=26910)
    if BUILD_REAL_NETWORK: real_auto.network_from_polygons(
        filepath="/Users/nicholasmartino/GoogleDrive/Geospatial/Databases/Hillside Quadra.gpkg",
        layer='land_blocks', scale_factor=0.84, buffer_radius=11, max_linters=0.40)

    for VERSION in VERSIONS:
        proxy = GeoBoundary(municipality=f'Hillside Quadra Proxy{VERSION}', crs=26910)
        if BUILD_PROXY_NETWORK: proxy.network_from_polygons(
            filepath=f"/Users/nicholasmartino/GoogleDrive/Geospatial/Databases/Hillside Quadra Proxy{VERSION}.gpkg",
            layer='land_parcels', scale_factor=0.80, buffer_radius=10, max_linters=0.25, remove_islands=False)

    if NETWORK_STATS:
        real_auto.centrality()
        rrep = real_auto.network_report()
        for VERSION in VERSIONS:
            proxy = GeoBoundary(municipality=f'Hillside Quadra Proxy{VERSION}', crs=26910)
            proxy.centrality()
        prep = proxy.network_report()
        print(rrep - prep)
        print("Done")
