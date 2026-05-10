import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


ox.settings.use_cache = True
ox.settings.cache_folder = "data/osmnx_cache"

def _first(list_or_scalar):
    if type(list_or_scalar) == list:
        return list_or_scalar[0]
    else:
        return list_or_scalar
    
def _priority_from_highway(highway):
    if (highway == 'motorway') or (highway == 'trunk') or (highway == 'primary'):
        return 1
    elif  (highway == 'secondary') or (highway == 'tertiary'): 
        return 2
    else:
        return 3


class RoadNetwork:

    def __init__(self, zone="midtown", lat=40.7580, lon=-73.9855, dist=1200):

        self.zone = zone

        if zone == "manhattan":
            self.G = ox.graph_from_place('Manhattan, New York, USA', network_type="drive")
        else:
            self.G = ox.graph_from_point((lat, lon), dist=dist, network_type="drive")
        self.G = ox.project_graph(self.G)
        self.G=  ox.add_edge_speeds(self.G)
        # self.G = ox.simplification.consolidate_intersections(self.G, tolerance=15, rebuild_graph=True, dead_ends=False)

        self.edges = list(self.G.edges(keys=True)) 
        self.nodes = list(self.G.nodes)

        self.E = len(self.edges) # number of edges on the graph
        self.N = len(self.nodes) # number of nodes on the graph

        self.edge_to_idx = {e: i for i, e in enumerate(self.edges)} # shape (E,)
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.next_options = {self.node_to_idx[n]: [] for n in self.nodes}
        for (u, v, k) in self.edges:
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            self.next_options[u_idx].append((v_idx, self.edge_to_idx[(u, v, k)]))

        # self.edges_at_node = self._get_edges_at_node()  # shape (N,)
        
        self.length   = np.array([self.G.edges[e]['length'] for e in self.edges], dtype=np.float32)# shape (E,)
        self.num_lanes  = np.array([_first(self.G.edges[e].get('lanes', '1')) for e in self.edges], dtype = np.int8) # shape (E,)
        self.highway  = [_first(self.G.edges[e].get('highway', 'unclassified')) for e in self.edges] # shape (E,)
        self.priority = np.array([_priority_from_highway(h) for h in self.highway], dtype=np.int8) # shape (E,)
        self.max_speed = np.array([self.G.edges[e].get('speed_kph') for e in self.edges])*1000 # shape (E,) in meters per hour

        xy = np.array([[self.G.nodes[n]['x'], self.G.nodes[n]['y']] for n in self.nodes], dtype=np.float32)
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        self.node_coords = (xy - xy_min) / (xy_max - xy_min)  # shape (N, 2), normalized to [0, 1]
        self.node_out_degree = np.array([self.G.out_degree(n) for n in self.nodes], dtype=np.int32)  # shape (N,)


        self.node_idxs_on_edges = []
        for u,v,_ in (self.edges):
            idx1 = self.node_to_idx[u]
            idx2 = self.node_to_idx[v]
            self.node_idxs_on_edges.append(np.array([idx1, idx2])) #appending 1D vector of shape (2,)
        self.node_idxs_on_edges = np.array(self.node_idxs_on_edges) #should be an array of shape (E, 2)

        self.M = np.zeros((self.N, self.E), dtype=np.float32) # M[n, e] = 1 if edge e is incident to node n, and 0 otherwise
        for e_idx in range(self.E):
            dst_node = self.node_idxs_on_edges[e_idx, 1]
            self.M[dst_node, e_idx] = 1.0

        self.max_degree = max(len(v) for v in self.next_options.values())


        self.snow = np.zeros(self.E, dtype = np.float32) # shape (E,)
        self.traffic = np.zeros(self.E, dtype = np.float32) # shape (E,)
        # self.plowed = np.zeros(self.E, dtype = np.float32) # shape (E,)

    # def _get_edges_at_node(self):

    #     edges_at_node = [] 

    #     for n in self.nodes:
    #         edges_at_current_node = set()
    #         for e in self.edges:
    #             if n in e:
    #                 edges_at_current_node.add(e)
            
    #         edges_at_node.append(edges_at_current_node)
    #     return edges_at_node
        

    def plot(self):

        nc = ox.plot.get_node_colors_by_attr(self.G, attr="y", cmap="plasma")
        fig, ax = ox.plot.plot_graph(
            self.G,
            node_color=nc,
            edge_linewidth=0.3,
            close = False,
            show = False
        )

        return fig, ax

    def plot_edge_heatmap(self, values):

        nonzero = values[values != 0]
        norm = mcolors.Normalize(vmin=nonzero.min(), vmax=nonzero.max())
        colors = cm.viridis(norm(values))
        colors[values == 0] = mcolors.to_rgba("red")

        fig, ax = ox.plot.plot_graph(
            self.G,
            edge_color=colors,
            node_size=0,
            edge_linewidth=0.5,
            show=False,
            close=False,
        )

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)

        return fig










