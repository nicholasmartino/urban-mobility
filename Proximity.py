import geopandas as gpd
import pandas as pd
from UrbanZoning.City.Network import Streets
from Morphology.ShapeTools import Analyst


class Proximity:
	def __init__(self, gdf, reference_gdf, radius=400):
		self.gdf = gdf
		self.reference_gdf = reference_gdf
		self.radius = radius

	def get_proximity(self):
		"""
		Summarizes elements from reference_gdf within a defined radius from the gdf
		:return:
		"""

		gdf = self.gdf.copy()
		ref_gdf = self.reference_gdf.copy()

		gdf['geometry'] = gdf.buffer(self.radius)

		gdf = Analyst(gdf, ref_gdf).spatial_join(operations=['sum', 'mean'])
		return gdf

	def test_get_proximity(self):
		assert type(self.gdf) == type(gpd.GeoDataFrame()), TypeError("Wrong data type for gdf parameter")
		assert type(self.reference_gdf) == type(gpd.GeoDataFrame()), TypeError("Wrong data type for reference_gdf parameter")
		assert 'geometry' in self.reference_gdf.columns, AssertionError("Geometry column not found in reference_gdf")
		assert self.gdf.crs == self.reference_gdf.crs, AssertionError(f"Base GeoDataFrame CRS ({self.gdf.crs}) "
		                                                              f"and reference GeoDataFrame CRS ({self.reference_gdf.crs}) are not the same")
		results = self.get_proximity()


if __name__ == '__main__':
	streets_gdf = gpd.read_file(f"/Volumes/Macintosh HD/Users/nicholasmartino/Google Drive/elementslab/main_st_streets_e0.geojson")
	Proximity(
		gdf=Streets(streets_gdf[streets_gdf['BikeLane'] == 1]).segmentize(),
	    reference_gdf=gpd.read_feather('/Volumes/SALA/Research/eLabs/50_projects/20_City_o_Vancouver/SSHRC Partnership Engage/'
	                                'Sandbox/shp/MainSt/Experiment/Mode Shares/'
	                                'ModeShares_sandbox_prcls_E0.shp_sandbox_prcls_E1.shp_bike.feather').loc[:, ['res_units', 'bike', 'geometry']].to_crs(26910)
	).test_get_proximity()
