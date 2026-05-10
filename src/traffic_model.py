import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import os
import pickle
import zipfile
import pyarrow as pa
import pyarrow.parquet as pq
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import igraph as ig
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import duckdb


REPO_ROOT = Path(__file__).parent.parent

# Module-level globals for parallel route workers.
# Relies on Linux fork() — workers inherit parent memory without pickling the graph.
_worker_igraph = None
_worker_osmid_to_ig_idx = None
_worker_ig_edge_to_rn_idx = None
_worker_E = None

def _process_chunk(chunk):
    counts = np.zeros((_worker_E, 168))
    for start, end, hr in chunk:
        src = _worker_osmid_to_ig_idx.get(start)
        tgt = _worker_osmid_to_ig_idx.get(end)
        if src is None or tgt is None:
            continue
        paths = _worker_igraph.get_shortest_paths(src, to=tgt, weights="travel_time", output="epath")
        if not paths or not paths[0]:
            continue
        for e_idx in paths[0]:
            rn_idx = _worker_ig_edge_to_rn_idx.get(e_idx)
            if rn_idx is None:
                continue
            counts[rn_idx, hr] += 1
    return counts


MANHATTAN_ZONES = [4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263]

class TrafficModel:

    def __init__(self, road_network = None):
        
        self.tlc_parquet_path =  REPO_ROOT / "data" / "processed" /  "tlc_nyc.parquet"
        self.tlc_csv_nodes_path =  REPO_ROOT / "data" / "processed" /  "tlc_nyc_nodes.csv"
        self.atvc_csv_path =  REPO_ROOT / "data" / "processed" /  "atvc_nyc.csv"

        self.raw_dir = REPO_ROOT / "data" /"raw" / "tlc_records"
        self.processed_dir = REPO_ROOT / "data" / "processed"

        self.traffic_counts = None

        self.rn = road_network
        self.E = road_network.E

    def step(self, t = 0):

        if self.traffic_counts is None:
            self.load_routes()

        return self.traffic_counts[:,t] 
 
    def load_routes(self, minutes_per_step=60):

        zone = self.rn.zone
        if zone == "manhattan":
            counts_path = self.processed_dir / "traffic_counts.pkl"
            if not os.path.exists(counts_path):
                self.build_routes()
            else:
                with open(counts_path, "rb") as f:
                    self.traffic_counts = pickle.load(f)
        else:
            counts_path = self.processed_dir / f"traffic_counts_{zone}.pkl"
            if not os.path.exists(counts_path):
                raise FileNotFoundError(
                    f"{counts_path} not found. Run: python scripts/convert_traffic_to_subarea.py --zone {zone} --lat <lat> --lon <lon> --dist <dist>"
                )
            with open(counts_path, "rb") as f:
                self.traffic_counts = pickle.load(f)

        steps_per_hr = 60 // minutes_per_step
        dirichlet_fractions = np.random.dirichlet(np.ones(steps_per_hr) * 10, size=(self.E, 168)).reshape(self.E, -1)
        self.traffic_counts = np.round(np.repeat(self.traffic_counts, steps_per_hr, axis=1) * dirichlet_fractions).astype(int)

    def build_routes(self):

        print("Constructing full NYC level graph of road network...")

        self.full_road_network = ox.graph_from_place(["Manhattan, New York, USA", "Brooklyn, New York, USA", "Queens, New York, USA", "Bronx, New York, USA"], network_type="drive")
        self.full_road_network = ox.project_graph(self.full_road_network, to_crs=self.rn.G.graph["crs"])
        self.full_road_network = ox.add_edge_speeds(self.full_road_network)
        self.full_road_network = ox.add_edge_travel_times(self.full_road_network)

        assert self.full_road_network.graph["crs"] == self.rn.G.graph["crs"], f"CRS mismatch: full={self.full_road_network.graph['crs']}, manhattan={self.rn.G.graph['crs']}"

        if (not os.path.exists(self.tlc_csv_nodes_path)):
            self.aggregate_tlc_trips()

        print('Building routes...')

        self.traffic_counts = np.zeros((self.rn.E, 168)) # shape (E, 168): number of edges x number of hrs in week

        con = duckdb.connect()
        rows = con.execute(f"SELECT start_node, end_node, pickup_datetime FROM read_csv_auto('{self.tlc_csv_nodes_path}')").fetchall()
        trips = [(r[0], r[1], pd.to_datetime(r[2]).dayofweek * 24 + pd.to_datetime(r[2]).hour) for r in tqdm(rows, desc="Parsing trips")]

        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        # chunk_size = max(1, len(trips) // n_workers)
        chunk_size = 100000

        print("Total number of trips is ",len(trips))
        chunks = [trips[i:i + chunk_size] for i in tqdm(range(0, len(trips), chunk_size), desc="Chunking trips")]


        ig_graph = ig.Graph.from_networkx(self.full_road_network)
        osmid_to_ig_idx = {v["_nx_name"]: v.index for v in tqdm(ig_graph.vs, desc="Building node index")}
        has_key = "key" in ig_graph.es.attributes()
        ig_edge_to_rn_idx = {}
        for e in tqdm(ig_graph.es):
            u_osmid = ig_graph.vs[e.source]["_nx_name"]
            v_osmid = ig_graph.vs[e.target]["_nx_name"]
            key = e["key"] if has_key else 0
            rn_idx = self.rn.edge_to_idx.get((u_osmid, v_osmid, key))
            if rn_idx is not None:
                ig_edge_to_rn_idx[e.index] = rn_idx

        global _worker_igraph, _worker_osmid_to_ig_idx, _worker_ig_edge_to_rn_idx, _worker_E
        _worker_igraph = ig_graph
        _worker_osmid_to_ig_idx = osmid_to_ig_idx
        _worker_ig_edge_to_rn_idx = ig_edge_to_rn_idx
        _worker_E = self.rn.E

        partial_counts = process_map(_process_chunk, chunks, max_workers=n_workers, desc="Building routes")

        self.traffic_counts = np.sum(partial_counts, axis=0)


        scaling = self.get_traffic_volume_scaling()

        self.traffic_counts = np.round(self.traffic_counts * scaling).astype(int)


        with open(self.processed_dir / "traffic_counts.pkl", "wb") as f:
            pickle.dump(self.traffic_counts, f)

    def aggregate_tlc_trips(self, 
                            months = ["January","February","December"], 
                            years = 2025, force_update = False):

        if ((not os.path.exists(self.tlc_parquet_path)) or force_update):
            self.create_tlc_dataset(months, years)

        self.load_zone_to_node_mapping()

        print("Assigning random start/end points to the trip within each TCL zone...")

        con = duckdb.connect()
        zones = ', '.join(str(z) for z in MANHATTAN_ZONES)
        total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{self.tlc_parquet_path}') WHERE PULocationID IN ({zones}) OR DOLocationID IN ({zones})").fetchone()[0]
        result = con.execute(f"SELECT pickup_datetime, PULocationID, DOLocationID FROM read_parquet('{self.tlc_parquet_path}') WHERE PULocationID IN ({zones}) OR DOLocationID IN ({zones})")
        cols = ['pickup_datetime', 'PULocationID', 'DOLocationID', 'start_node', 'end_node']
        first = True
        batch_size = 100000
        with tqdm(total=total, desc="Assigning nodes") as pbar:
            while True:
                batch = result.fetchmany(batch_size)
                if not batch:
                    break
                pbar.update(len(batch))
                rows = [
                    (ts, pu, do_, np.random.choice(self.zone_to_node_lookup[pu]), np.random.choice(self.zone_to_node_lookup[do_]))
                    for ts, pu, do_ in batch
                    if self.zone_to_node_lookup.get(pu) and self.zone_to_node_lookup.get(do_)
                ]
                if rows:
                    pd.DataFrame(rows, columns=cols).to_csv(
                        self.tlc_csv_nodes_path, mode='w' if first else 'a', header=first, index=False)
                    first = False

        print(f"Saved Processed .csv at {self.tlc_csv_nodes_path} with added random PU/DO points")

    

    def load_zone_to_node_mapping(self):

        print("Creating TLC zone to graph node mapping...")
            
        if os.path.exists(self.processed_dir / "tlc_zone_to_node.pkl"):
            with open(self.processed_dir / "tlc_zone_to_node.pkl", "rb") as f:
                self.zone_to_node_lookup = pickle.load(f)

        else:

            zone_shapefile = self.raw_dir / 'taxi_zones' / 'taxi_zones.shp'
            
            if not os.path.exists(zone_shapefile):
                zone_shapefile_zip = self.raw_dir / "taxi_zones.zip"
                if not os.path.exists(zone_shapefile_zip):
                    try:
                        urllib.request.urlretrieve('https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip', zone_shapefile_zip)
                        print(f"Downloaded {zone_shapefile_zip}")
                    except Exception as e:
                        print(e)

                with zipfile.ZipFile(zone_shapefile_zip, 'r') as z:
                    z.extractall(self.raw_dir)

            zones = gpd.read_file( zone_shapefile, engine="pyogrio")
            zones = zones.to_crs(self.rn.G.graph["crs"])

            self.zone_to_node_lookup = {}

            for index, row in zones.iterrows():
                self.zone_to_node_lookup[row["LocationID"]] = []

            for node in tqdm(self.full_road_network.nodes):
                x_coord = self.full_road_network.nodes[node]["x"]
                y_coord = self.full_road_network.nodes[node]["y"]

                for index, row in zones.iterrows():
                    if row["geometry"].contains(Point(x_coord, y_coord)):
                        self.zone_to_node_lookup[row["LocationID"]].append(node)

            with open(self.processed_dir / "tlc_zone_to_node.pkl", "wb") as f:
                pickle.dump(self.zone_to_node_lookup, f)

    def create_tlc_dataset(self, months = ["January"], years = np.arange(2021, 2026)):

        print("Creating TCL dataset...")

        if type(months) == str:
            self.months = [months]
        else:
            self.months = months

        if type(years) == int:
            self.years = [years]
        else:
            self.years = years

        # Create an empty dataframe 
        used_columns = [
            'pickup_datetime',
            'dropoff_datetime', 
            'PULocationID', 
            'DOLocationID', 
            'trip_miles',
            'trip_time'
            ]
        

        filters = [
            [("PULocationID", "in", MANHATTAN_ZONES)],
            [("DOLocationID", "in", MANHATTAN_ZONES)],
        ] # this disrgards trips from bronx-brooklyn that route through manhattan

        tmp_path = self.tlc_parquet_path.with_name("tlc_nyc_tmp.parquet")

        if tmp_path.exists():
            tmp_path.unlink()

        writer = None

        for year in tqdm(self.years):
            for month in self.months:

                save_path = self.fetch_tlc_trip_records(month=month, year=year)
                df = pd.read_parquet(save_path, columns=used_columns, filters=filters, memory_map=True)
                table = pa.Table.from_pandas(df, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, table.schema)

                writer.write_table(table.cast(writer.schema))

        writer.close()

        tmp_path.rename(self.tlc_parquet_path)
        
        print(f"Saved processed parquet at {self.tlc_parquet_path}")

    def fetch_tlc_trip_records(self, month = "January", year = 2025):

        print("Fetching TCL trip records...")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        month_mapping = {
            "January": "01",
            "February": "02",
            "March": "03",
            "April": "04",
            "May": "05",
            "June": "06",
            "July": "07",
            "August": "08",
            "September": "09",
            "October": "10",
            "November": "11",
            "December": "12"
        }

        try:
            month = month_mapping[month]
        except:
            raise KeyError(f"{month} is invalid name for a month, please use a valid name!")
        
        trip_type = 'fhvhv'
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{trip_type}_tripdata_{year}-{month}.parquet"
        fname = f"{year}-{month}.parquet"
        save_path = self.raw_dir / fname

        if save_path.exists():
            pass
        else:
            try:
                urllib.request.urlretrieve(url, save_path)
                print(f"Downloaded {fname}")
            except Exception as e:
                print(e)

        return save_path
    
    def fetch_atvc_data(self):

        print("Fetching ATVC data...")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        url =  "https://data.cityofnewyork.us/api/views/7ym2-wayt/rows.csv?accessType=DOWNLOAD"

        try:
            urllib.request.urlretrieve(url,  self.atvc_csv_path)
            print(f"Downloaded {self.atvc_csv_path}")
        except Exception as e:
            print(e)


    def build_lion_to_osm_lookup(self, max_distance_m=15):

    
        cache_path = self.processed_dir / "lion_to_osm.pkl"
        if cache_path.exists():
            print("Loading lion-to-osm lookup table...")
            with open(cache_path, "rb") as f:
                self.lion_to_osm = pickle.load(f)
            return self.lion_to_osm
        
        print("Building lion-to-osm lookup table...")

        lion_dir = self.raw_dir.parent / "lion"
        lion_zip = lion_dir / "lion.zip"
        lion_dir.mkdir(parents=True, exist_ok=True)

        if not lion_zip.exists():
            try:
                # urllib.request.urlretrieve("https://data.cityofnewyork.us/download/2v4z-66xt/application%2Fzip", lion_zip)
                urllib.request.urlretrieve("https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/lion/nyc_lion14d.zip", lion_zip) #version 14
            except Exception as e:
                print(e)

        with zipfile.ZipFile(lion_zip, "r") as z:
            z.extractall(lion_dir)

        lion_shp = next(lion_dir.glob("**/lion.gdb"))
        lion = gpd.read_file(lion_shp, layer="lion", engine="pyogrio")
        lion = lion[lion["LBoro"] == 1].to_crs(self.rn.G.graph["crs"])

        osm_edges = ox.graph_to_gdfs(self.rn.G, nodes=False, edges=True).reset_index()

        joined = gpd.sjoin_nearest(
            lion[["SegmentID", "geometry"]],
            osm_edges[["u", "v", "key", "geometry"]],
            how="left",
            max_distance=max_distance_m,
        )

        self.lion_to_osm = {
            int(row["SegmentID"]): self.rn.edge_to_idx[(int(row["u"]), int(row["v"]), int(row["key"]))]
            for _, row in joined.iterrows()
            if not pd.isna(row.get("u"))
            and (int(row["u"]), int(row["v"]), int(row["key"])) in self.rn.edge_to_idx
        }

        with open(cache_path, "wb") as f:
            pickle.dump(self.lion_to_osm, f)

        return self.lion_to_osm


    def get_traffic_volume_scaling(self, update_traffic_count_points = False, num_sample_points =10):

        cache_path = self.processed_dir / "selected_traffic_counts_from_atvc.pkl"

        if (os.path.exists(cache_path)) and (not update_traffic_count_points):
            with open(cache_path, "rb") as f:
                self.traffic_counts_selected = pickle.load(f)

        else:

            if not os.path.exists(self.atvc_csv_path):
                self.fetch_atvc_data()

            self.build_lion_to_osm_lookup()

            atvc_df_full = pd.read_csv(self.atvc_csv_path)
            atvc_df = atvc_df_full[(atvc_df_full["Yr"].astype(int).isin(range(2020, 2026))) & (atvc_df_full["Boro"] == "Manhattan")] #TODO: fix the years range here to reference class variable
    
            #Assume we take 10 points in mahnattan:
            traffic_counts_selected = np.zeros((num_sample_points, 168)) # shape (num_sample_points, num_hr_week) = (10, 168)
            lion_to_osm_selected = np.zeros(num_sample_points, dtype = np.int64) #shape = (num_sample_points,) = (10, )

            remaining_segments = list(set(atvc_df["SegmentID"]))
            np.random.shuffle(remaining_segments)

            segments_and_directions = []
            for segment in remaining_segments:
                if len(segments_and_directions) == num_sample_points:
                    break

                osm_idx = self.lion_to_osm.get(int(segment))
                if osm_idx is None:
                    continue

                direction = np.random.choice(list(set(atvc_df[atvc_df["SegmentID"] == segment]["Direction"])))
                lion_to_osm_selected[len(segments_and_directions)] = osm_idx
                segments_and_directions.append((segment, direction))

            for i, (segment, direction) in enumerate(segments_and_directions):

                seg_df = atvc_df[(atvc_df["SegmentID"] == segment) & (atvc_df["Direction"] == direction)]

                hourly = seg_df.groupby(["Yr", "M", "D", "HH"], as_index=False)["Vol"].sum()
                dates = pd.to_datetime(pd.DataFrame({
                    "year": hourly["Yr"].astype(int),
                    "month": hourly["M"].astype(int),
                    "day": hourly["D"].astype(int),
                }))
                hr_of_week = dates.dt.dayofweek * 24 + hourly["HH"].astype(int)
                mean_hourly = hourly["Vol"].groupby(hr_of_week).mean()
                for hr, vol in mean_hourly.items():
                    traffic_counts_selected[i, int(hr)] = vol


            self.traffic_counts_selected = {"counts" : traffic_counts_selected, "osm_mapping" : lion_to_osm_selected}

            with open(cache_path, "wb") as f:
                pickle.dump(self.traffic_counts_selected, f)

        scaling_factor = np.zeros(168)
        for i,osm_segment in enumerate(self.traffic_counts_selected["osm_mapping"]):
            scaling_factor +=  (self.traffic_counts_selected["counts"][i, :] + 0.1) / (self.traffic_counts[osm_segment, :] + 0.1) / len(self.traffic_counts_selected["osm_mapping"])

        return scaling_factor