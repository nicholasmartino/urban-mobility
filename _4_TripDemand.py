import pandas as pd
import numpy as np
import geopandas as gpd
import pandana.network as pdna

# States: travel to destination, travel home and not travel
# states = ["travel_out", "travel_home", "not_travel"]

class Environment:
    def __init__(self, origins_gdf, destinations_gdf, trip_diary, trip_rate=3.3):

        # Load trip rate
        self.trip_rate = trip_rate

        # Load trip diary
        self.trip_length = trip_diary['length']
        self.diary = trip_diary.drop(['length'], axis=1)

        # Load origins and destinations
        self.origins = origins_gdf
        self.destinations = destinations_gdf
        return

    def trips_by_purpose(self, purposes='all'):
        """
        Return the percentage of trips by purpose

        :param purposes: a list with one or more column names from the trip_diary DataFrame
        :return: the filtered trip diary
        """

        df = self.diary / self.diary.sum(axis=0)
        if purposes == 'all': return df
        else: return df.loc[:, purposes]

    def trips_by_period(self, periods='all'):
        """
        Return the percentage of trips by period

        :param periods: an index in the trip_diary DataFrame
        :return: the filtered trip diary
        """

        df = (self.diary.transpose() / self.diary.transpose().sum(axis=0).transpose()).transpose()
        df.index.name = None
        if periods == 'all': return df
        else: return df.loc[periods, :]



class Commuter:
    def __init__(self, environment, periods=24):
        self.env = environment
        self.periods = periods
        self.trips = []
        self.prob = {}
        return

    def forecast_trip(self):
        cur_state = 'home'

        # Check if current number of trips is smaller than the trip rate
        if len(self.trips) < self.env.trip_rate:

            # Iterate over time periods
            for per in range(self.periods):
                per = per+1
                self.prob[per] = {}

                # Get probability of travelling in this period
                by_per = self.env.trips_by_period(per)

                # Iterate over purposes
                for pur in by_per.index:
                    if cur_state == 'home' and pur == 'home': pass
                    else:
                        by_pur = self.env.trips_by_purpose(pur)

                        # Calculate the probability of travelling to specific purposes
                        self.prob[per][pur] = by_pur[per] * by_per.at[pur]

                # Assign probability of staying where it is
                self.prob[per]['stay'] = 1 - sum(self.prob[per].values())

                # Getting next direction
                move = np.random.choice(list(self.prob[per].keys()), replace=True, p=list(self.prob[per].values()))
                if move == 'stay': pass

                # Chose destination to travel if it decide to move
                else:
                    # Get distance to all possible destinations within this period's trip length (+20%)
                    self.env.trip_length[per]

                    orig_n = len(sample_gdf.geometry)
                    print(
                        f'\n> Network analysis for {orig_n} geometries at {service_areas} buffer radius in {self.city_name}')

                    # Load data
                    nodes = self.nodes
                    edges = self.links
                    print(nodes.head(3))
                    print(edges.head(3))
                    nodes.index = list(nodes['osmid'].astype(int))
                    edges["from"] = pd.to_numeric(edges["from"])
                    edges["to"] = pd.to_numeric(edges["to"])

                    # Probability of destinations = area of destination / distance from home
                    self.env.destinations['purpose'] = self.env.destinations['area']

                    des = np.random.choice(list(self.env.destinations), replace=True, p=list(self.prob[per].values()))
                    cur_state = move
                    self.trips.append(move)

        return


if __name__ == '__main__':
    ddf = gpd.read_file('/Volumes/Samsung_T5/Databases/Sandbox/Sunset/Sunset Sandbox.gpkg', layer='land_parcels_e0')
    ddf = ddf[(ddf["LANDUSE"] == 'MX') | (ddf["LANDUSE"] == 'CM')]
    env = Environment(
        trip_diary=pd.read_csv('/Volumes/Samsung_T5/Databases/TransLink/TripDiary2017.csv', index_col='time'),
        origins_gdf=gpd.read_file('/Volumes/Samsung_T5/Databases/Sandbox/Sunset/Sunset Sandbox.gpkg', layer='land_parcels_e0'),
        destinations_gdf=ddf,
        # destinations_gdf=gpd.read_file('/Volumes/Samsung_T5/Databases/Metro Vancouver, British Columbia.gpkg', layer='land_assessment_parcels')
    )

    for period in range(24):
        # period = period+1
        # df = env.trips_by(periods=[period])
        # prob_travel_home = df.loc[:, 'home']
        # prob_travel_out = df.loc[:, df.columns != 'home']

        for i in range(1500):
            comm = Commuter(env)
            comm.forecast_trip()
            prob_travel_home = prob_travel_home * comm.outside
            Commuter(env).forecast_trip()
