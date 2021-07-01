import os
import geopandas as gpd

def read_gdfs(f_path, files):
    gdfs = {}
    for file in files:
        if file is not None:
            file_type = file.split('.')[len(file.split('.')) - 1]
            if file_type == 'feather':
                gdf = gpd.read_feather(f'{f_path}/{file}').to_crs(26910)
            else:
                gdf = gpd.read_file(f'{f_path}/{file}').to_crs(26910)
            gdfs[file] = gdf
    return gdfs

def export_multi(gdf, formats, directory='', file=''):
    if not os.path.exists(directory): os.mkdir(directory)
    for f in formats:
        if f == '.feather': gdf.to_feather(f'{directory}/{file}.feather')
        if f == '.geojson': gdf.to_file(f'{directory}/{file}.geojson', driver='GeoJSON')
        if f == '.shp': gdf.to_file(f'{directory}/{file}.shp')
    return
