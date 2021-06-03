import geopandas as gpd
from Morphology.NetworkTools import Network
from UrbanZoning.City.Network import Streets


class Proximity:
	def __init__(self, gdf, reference_gdf, network):
		self.gdf = gdf
		self.reference_gdf = reference_gdf
		self.network = network

	def get_proximity(self):
		self.network.build()
		return

	def test_get_proximity(self):
		assert type(self.gdf) == type(gpd.GeoDataFrame()), TypeError("Wrong data type for gdf parameter")
		assert type(self.reference_gdf) == type(gpd.GeoDataFrame()), TypeError("Wrong data type for reference_gdf parameter")
		assert self.network.__class__.__name__ == 'Network', TypeError("Wrong data type for network parameter")
		assert self.gdf.crs == self.reference_gdf.crs, AssertionError("Base GeoDataFrame CRS and reference GeoDataFrame CRS are not the same")
		results = self.get_proximity()


if __name__ == '__main__':
	streets_gdf = gpd.read_file(f"/Volumes/Macintosh HD/Users/nicholasmartino/Google Drive/elementslab/main_st_streets_e0.geojson")
	Proximity(
		gdf=gpd.read_feather('/Volumes/Samsung_T5/Databases/Sandbox/Main Street/Network/Main Street Sandbox_mob_sandbox_prcls_E0.shp_na.feather'),
	    reference_gdf=gpd.read_feather('/Volumes/Samsung_T5/Databases/Sandbox/Main Street/Network/Main Street Sandbox_mob_sandbox_prcls_E2.shp_na.feather'),
		network=Network(
			edges_gdf=streets_gdf,
			nodes_gdf=Streets(streets_gdf).extract_intersections())
	).test_get_proximity()
