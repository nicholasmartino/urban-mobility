import os

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

