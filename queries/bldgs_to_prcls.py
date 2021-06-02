from Sandbox import bldgs_to_prcls
import geopandas as gpd

directory = '/Volumes/ELabs/50_projects/20_City_o_Vancouver/SSHRC Partnership Engage/Sandbox/shp/elementslab/Version_2'
cg_bld = gpd.read_file(f'{directory}/Coarse_Grain_bldgs.shp')
cg_pcl = gpd.read_file(f'{directory}/Coarse_Grain_prcls.shp')
ol_bld = gpd.read_file(f'{directory}/Open_Low_Density_bldgs.shp')
ol_pcl = gpd.read_file(f'{directory}/Open_Low_Density_prcls.shp')

cg_pcl = bldgs_to_prcls(cg_bld, cg_pcl)
ol_bld = bldgs_to_prcls(ol_bld, ol_pcl)

cg_pcl['cell_type'] = 'Coarse_Grain'
cg_pcl.to_file(f'{directory}/Coarse_Grain_prcls.shp')

ol_pcl['cell_type'] = 'Open_Low_Density'
ol_pcl.to_file(f'{directory}/Open_Low_Density_prcls.shp')
