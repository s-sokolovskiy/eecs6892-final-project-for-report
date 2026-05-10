import numpy as np
import matplotlib.pyplot
from src.road_network import RoadNetwork
from src.traffic_model import TrafficModel
from src.snow_model import SnowModel


class WinterStorm:

    def __init__(self, start_hour = 0, road_network = None, tick_size = 1, weather_sampling_mode = "markov" ):


        self.road_network = road_network
        self.start_tick = start_hour * 60 // tick_size
        self.current_tick =  start_hour * 60 // tick_size
        self.tick_size = tick_size

        self.snow_model = SnowModel(road_network=self.road_network, mode=weather_sampling_mode)
        self.traffic_model = TrafficModel(road_network=self.road_network)

        self.reset()

    def reset(self, start_hour = None):

        if start_hour is not None:
            self.start_tick = start_hour * 60 // self.tick_size

        self.current_tick = 0

        self.snow_accumulation = np.zeros(self.road_network.E)
        self.current_traffic_demand = np.zeros(self.road_network.E)
        self.snow_cleared_tick = 0.0
        self.reward_tick = 0.0

        self.snow_model.load_storm(minutes_per_step=self.tick_size) #storm duration in ticks
        # print("Loaded Snow")
        self.traffic_model.load_routes(minutes_per_step=self.tick_size)
        # print("Loaded Traffic")


    def step(self, plows):

        added_snow, done = self.snow_model.step(t=self.current_tick)
        self.snow_accumulation += added_snow

        ticks_per_week = 168 * 60 // self.tick_size
        self.current_traffic_demand = self.traffic_model.step(t= int((self.current_tick + self.start_tick) % ticks_per_week))
        self.snow_cleared_tick = 0.0
        self.reward_tick = 0.0

        plows = np.array([self._plow_step(plows[i]) for i in range(len(plows))])

        self.current_tick += 1

        return self.snow_accumulation, self.current_traffic_demand, plows, self.reward_tick, done


    def _plow_step(self, plow):
        """
        Plow is the dict of the following form:
        {
        "start_node" = int, starting node of the edge where it currently is 
        "end_node" = int, ending node of the edge where it currently is 
        "edge" =int, index if the edge where the plow currently is 
        "position" = float in [0,1] that describes relative position of the plow on the edge
        "mode" = str, 'driving' or 'plowing'
        }
        """

        if plow[4] == 0.0: #simple driving
            traversed_distance = (self.tick_size / 60) * self.road_network.max_speed[int(plow[2])] /self.road_network.length[int(plow[2])]
            plow[3] = np.minimum(1.0, plow[3] + traversed_distance)

        elif plow[4] == 1.0: #plowing

            traversed_distance = (self.tick_size / 60) * 25000 /self.road_network.length[int(plow[2])] #assuming fixed speed of plowing of 25000 meter per hour
            plow[3] = np.minimum(1.0, plow[3] + traversed_distance)

            cleared_snow = (1 / self.road_network.num_lanes[int(plow[2])]) * traversed_distance * self.snow_accumulation[int(plow[2])] 
            self.snow_accumulation[int(plow[2])] = np.maximum(0.0, self.snow_accumulation[int(plow[2])] - cleared_snow)
            self.snow_cleared_tick += cleared_snow
            self.reward_tick += cleared_snow *  self.current_traffic_demand[int(plow[2])] * 1e-4
        
        return plow


    def plot(self):
        
        pass