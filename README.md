# Mobility Mode Predictor

## Description
Applies a random forest model to predict mobility mode shares, 
mode shifts and GHG emissions for urban parcels in different
urban form future scenarios.

![](https://raw.githubusercontent.com/nicholas-martino/UrbanMobility/master/images/Dash_Experiments.png "Scenario comparison dashboard")

## Analyzing urban form
Urban form data (explanatory) is aggregated using network analysis tools from [Pandana](https://github.com/UDST/pandana). The data was trained using data from Metro Vancouver, Canada.
* Transit Frequency, downloaded from [OpenMobilityData](https://transitfeeds.com/p/translink-vancouver/29);
* Commercial Land Use, inferred from [OpenStreetMap](https://www.openstreetmap.org/) data and downloaded using [OSMnx](https://github.com/gboeing/osmnx);
* Population Density, downloaded from [Statistics Canada](https://www150.statcan.gc.ca/n1/pub/92-195-x/2011001/other-autre/pop/pop-eng.htm);
* Number of Dwellings, also downloaded from [Statistics Canada](https://www12.statcan.gc.ca/census-recensement/2016/ref/guides/001/98-500-x2016001-eng.cfm). 

## Predicting mode share
The predictive model was developed using the [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) 
object from Scikit-Learn. Training algorithm can be downloaded from [GeoLearning](https://github.com/nicholas-martino/GeoLearning/blob/master/_Mobility.py)
The trained model can be found at the regression folder.

## Calculating mode shifts
When more than one urban form layer is analyzed, mode shifts results can be calculated and displayed in a [Plotly](https://github.com/plotly/dash) dashboard.

## License
[![cc-by-image](https://i.creativecommons.org/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)