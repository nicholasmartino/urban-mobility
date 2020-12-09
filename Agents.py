import pandas as pd
import geopandas as gpd
from graph_tool.all import Graph
from graph_tool.topology import shortest_path
from shapely.ops import nearest_points
import numpy as np


class Environment:
    def __init__(self, streets, origins_gdf, destinations_gdf, trip_diary, trip_rate=3.3):

        # Load trip rate
        self.trip_rate = trip_rate

        # Load trip diary
        self.trip_length = trip_diary['length']
        self.diary = trip_diary.drop(['length'], axis=1)

        # Load origins and destinations
        self.origins = origins_gdf
        self.destinations = destinations_gdf

        # Load OSM network
        self.streets = streets
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
    """Commuter (to work)"""
    def __init__(self, environment, mode, block_id, income, periods=24):
        self.env = environment
        self.periods = periods
        self.trips = []
        self.prob = {}
        self.mode = mode
        self.block_id = block_id
        self.income = income
        self.home = True
        return

    def trip_length_from_mode(self):
        """
        :return: Length of trip to work based on mode choice and time period
        """

        # Average lengths by time period from TransLink Trip Diary 2017
        lengths = {
            'walk': [0, 0, 0, 0, 0, 1, 1.1, 1.2, 1.2, 0.9, 0.8, 0.8, 0.6, 0.5, 0.6, 0.8, 0.6, 0.8, 0.7, 0, 0, 0, 0, 0],
            'bike': [0, 0, 0, 0, 0, 9.7, 8.5, 7.4, 6.2, 4.4, 3.7, 4.9, 4.3, 3.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'bus': [0, 0, 0, 0, 27, 23.2, 21.9, 17.9, 14.1, 12.2, 15.4, 14.3, 11.5, 13.6, 9.7, 11, 12.4, 13.2, 10.5, 0, 0, 0, 0, 0],
            'drive': [0, 0, 0, 0, 20.1, 22.6, 19.2, 15.5, 12.6, 13.6, 12.8, 12.4, 10.7, 11, 11.6, 10.5, 10.3, 11.8, 11.2, 8.6, 10.9, 14, 15.2, 0]
         }
        lengths = pd.DataFrame(lengths)

        # Iterate over time periods
        for per in range(self.periods):
            if self.home:
                per = per+1
                self.prob[per] = {}

                # Get probability of travelling to work in this period
                self.prob[per]['move'] = self.env.trips_by_period(per)['work_univ']

                # Assign probability of staying where it is
                self.prob[per]['stay'] = 1 - self.prob[per]['move']

                # Getting next direction
                move = np.random.choice(list(self.prob[per].keys()), replace=True, p=list(self.prob[per].values()))

                # Get trip length if agent decides to move
                if move == 'stay':
                    return 0
                else:

                    # Get average length given the period and the mode
                    ave_length = lengths.at[per, self.mode]

                    # des = np.random.choice(list(self.env.destinations), replace=True, p=list(self.prob[per].values()))
                    # cur_state = move

                    self.trips.append(move)
                    self.home = not self.home
                    return ave_length

    def path_from_income(self):
        """
        :return: GeoDataFrame representing path of commuter to work
        """

        # Get jobs within range of income of commuter
        streets = self.env.streets.copy().reset_index(drop=True)
        origins = self.env.origins.copy()
        destinations = self.env.destinations.copy()
        destinations = destinations[(destinations['salary_n'] > self.income[0]) & (destinations['salary_n'] < self.income[1])]

        # Get shortest path to one random destination
        osm_g = Graph(directed=False)
        indices = {}
        if len(destinations) > 0:

            # Add vertices to graph
            for i, osm_id in enumerate(list(streets['from']) + list(streets['to'])):
                v = osm_g.add_vertex()
                v.index = int(i)
                indices[osm_id] = i

            # Add edges to graph
            for i in list(streets.index):
                o_osm = streets.at[i, 'from']
                d_osm = streets.at[i, 'to']
                osm_g.add_edge(indices[o_osm], indices[d_osm])

            # Randomly choose destination
            destination = destinations.loc[np.random.choice(list(destinations.index)), :]

            # Randomly choose origin parcel based on block id
            origins = origins[(origins['Landuse'].isin(['MFH', 'MFL', 'SFA', 'SFD', 'MX'])) & (origins['index_block'] == self.block_id)].reset_index(drop=True)
            if len(origins) > 0:
                origin = origins.loc[np.random.choice(list(origins.index)), :]

                # Get closest street of origin and destination
                osm_origin = streets[streets.centroid == nearest_points(origin['geometry'], streets.centroid.unary_union)[1]]
                osm_destination = streets[streets.centroid == nearest_points(destination['geometry'], streets.centroid.unary_union)[1]]

                # Calculate shortest path
                def keys(dictA, value):
                    return list(dictA.keys())[list(dictA.values()).index(value)]

                path = shortest_path(osm_g, indices[osm_origin['from'].values[0]], indices[osm_destination['to'].values[0]])[1]
                path_gdf = pd.concat([streets[(streets['from'] == keys(indices, int(edge.source()))) & (streets['to'] == keys(indices, int(edge.target())))] for edge in path])

                return path_gdf
            else: return None
        else: return None
