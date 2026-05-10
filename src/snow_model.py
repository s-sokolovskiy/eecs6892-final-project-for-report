import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import urllib.request
import os
from tqdm import tqdm
import pickle

REPO_ROOT = Path(__file__).parent.parent

class SnowModel:

    def __init__(self, road_network = None, mode = 'replay', sampling_stations = "Central Park", start_year = 2005, end_year = 2025):
        
        self.isd_csv_path =  REPO_ROOT / "data" / "processed" /  "isd_nyc.csv"
        self.storm_csv_path =  REPO_ROOT / "data" / "processed" /  "storms.csv"
        self.markov_model_path = REPO_ROOT / "data" / "processed" / "markov.pkl"
        self.sampling_stations = sampling_stations
        self.start_year = start_year
        self.end_year = end_year

        self.rn = road_network
        self.E = road_network.E

        self.precipitation_per_step = None
        self.t = 0


        if (mode == 'markov') or (mode == 'replay'):
            self.mode = mode    
        else:
            raise ValueError(f"{mode} mode is invalid! Please use 'markov' or 'replay'!")
        

    def load_storm(self, minutes_per_step = 15): 

        if self.mode == 'replay':

            print("Storm sampling mode set to 'replay'")

            if (not os.path.exists(self.storm_csv_path)):
                self.identify_storm()

            storm_df = pd.read_csv(self.storm_csv_path)
            chosen_storm_idx = np.random.randint(len(storm_df))
            hr_precipitation = np.array(eval(storm_df.iloc[chosen_storm_idx]["Precipitation by Hour"]))
            
        elif self.mode == 'markov':

            # print("Storm sampling mode set to 'markov'")

            hr_precipitation = self.sample_markov()

        storm_buffer = np.tile(np.repeat(hr_precipitation, (60 // minutes_per_step)), (self.E, 1))
        dirichlet_fractions = np.random.dirichlet(np.ones(60 // minutes_per_step) * 10, size=(self.E, len(hr_precipitation))).reshape(self.E, -1)

        self.precipitation_per_step = storm_buffer * dirichlet_fractions * self.rn.length[:, None] * self.rn.num_lanes[:, None] # shape (E, T)
        self.t = 0

        # return np.shape(self.precipitation_per_step)[1]
        
    def step(self, t = None):

        if self.precipitation_per_step is None:
            self.load_storm()

        storm_length = np.shape(self.precipitation_per_step)[1]
        episode_length = storm_length + int(0.2 * storm_length)

        if t < storm_length:
            self.t = t
            return self.precipitation_per_step[:,self.t], False
        elif t < episode_length:
            return np.zeros_like(self.precipitation_per_step[:,-1]), False
        else:
            return np.zeros_like(self.precipitation_per_step[:,-1]), True


    def reset(self):
        self.t = 0
        self.load_storm()

    def train_markov(self):

        print("Building Markov Model for Winter Storms...")

        if (not os.path.exists(self.storm_csv_path)):
                self.identify_storm()

        storms = pd.read_csv(self.storm_csv_path)

        # 1) Build Transition Matrix
        all_hourly = np.concatenate([np.array(eval(s)) for s in storms["Precipitation by Hour"]])
        nonzero = all_hourly[all_hourly > 0]
        q33, q66 = np.quantile(nonzero, [0.33, 0.66])

        def to_state(x):
            # 4 states: 0=none, 1=light (0 < x <= q33), 2=moderate (q33 < x <= q66), 3=heavy (x > q66)
            if x == 0:
                return 0
            return int(np.searchsorted([q33, q66], x, side="left")) + 1

        counts = np.zeros((4,4))
        for s in storms["Precipitation by Hour"]:
            states = np.array([to_state(x) for x in np.array(eval(s))])
            for s_t, s_next in zip(states[:-1], states[1:]):
                counts[s_t, s_next] += 1

        self.P = (counts + 0.5) / (counts + 0.5).sum(axis=1, keepdims=True)


        # 2) Empirical distribution of mm/hr within each state
        self.empirical = [
            np.array([0.0]),
            nonzero[(nonzero > 0)   & (nonzero <= q33)],
            nonzero[(nonzero > q33) & (nonzero <= q66)],
            nonzero[(nonzero > q66)],
        ]


        # 3) Empirical distribution of initial states
        init_states = [to_state(np.array(eval(s))[0]) for s in storms["Precipitation by Hour"]]
        self.pi0 = np.bincount(init_states, minlength=4).astype(float)
        self.pi0 /= self.pi0.sum()

        # 4) Empirical Distribution for storm durations
        self.durations = np.array([len(eval(s)) for s in storms["Precipitation by Hour"]])


        with open(self.markov_model_path, "wb") as f:
            pickle.dump({
                "P": self.P,
                "pi0": self.pi0,
                "durations": self.durations,
                "empirical": self.empirical,
            }, f)

        self._validate_markov(storms)


    def sample_markov(self):

        if (not os.path.exists(self.markov_model_path)):
            self.train_markov()
        elif not hasattr(self, "P"):
            with open(self.markov_model_path, "rb") as f:
                cache = pickle.load(f)

            self.P = cache["P"]
            self.empirical = cache["empirical"]
            self.pi0 = cache["pi0"]
            self.durations = cache["durations"]

        s = np.random.choice(4, p=self.pi0)
        duration = int(np.random.choice(self.durations))
        
        hr_precipitation = np.empty(duration)
        for h in range(duration):
            hr_precipitation[h] = np.random.choice(self.empirical[s])  
            s = np.random.choice(4, p=self.P[s]) 

        return hr_precipitation


    def _validate_markov(self, storms):

        synthetic = [self.sample_markov() for _ in range(1000)]
        historical = [np.array(eval(s)) for s in storms["Precipitation by Hour"]]

        print("Total accumulation:")
        print("  hist:", np.percentile([s.sum() for s in historical], [25, 50, 75]))
        print("  syn :", np.percentile([s.sum() for s in synthetic],  [25, 50, 75]))

        print("Peak intensity:")
        print("  hist:", np.percentile([s.max() for s in historical], [25, 50, 75]))
        print("  syn :", np.percentile([s.max() for s in synthetic],  [25, 50, 75]))


    def identify_storm(self, force_download = False):

        print("Building hisotry of snow storms in NYC...")

        if ((not os.path.exists(self.isd_csv_path)) or force_download):
            self.fetch_data(self.sampling_stations, self.start_year, self.end_year)

        storm_df = pd.DataFrame(columns=["Start Datetime", "End Datetime", "Duration", "Precipitation by Hour", "Temp by Hour"])
        full_df = pd.read_csv(self.isd_csv_path, parse_dates=["datetime"])

        detected_storm = False
        snow_acc = []
        temp = []

        for index, row in tqdm(full_df.iterrows()):

            if (row["temp"] < 1.0) and (row["p1h"] > 0):
                if not detected_storm:

                    detected_storm = True

                    if not storm_df.empty:
                        previous_storm = storm_df.iloc[-1]

                        if (row["datetime"] - previous_storm["End Datetime"]).total_seconds() <= 7200:
                            start_dt =  previous_storm["Start Datetime"]
                            snow_acc = list(previous_storm.at["Precipitation by Hour"])     
                            temp =  list(previous_storm.at["Temp by Hour"])     
                            storm_df.drop(storm_df.index[-1], inplace=True)

                            gap_hours = int((row["datetime"] - previous_storm["End Datetime"]) / pd.Timedelta(hours=1)) - 1

                            for offset in range(gap_hours, 0, -1):
                                gap_row = full_df.iloc[index - offset]
                                snow_acc.append(0.0)
                                temp.append(gap_row["temp"])

                        else:
                            start_dt = row["datetime"]
                            snow_acc.append(row["p1h"])
                            temp.append(row["temp"])
                    else:
                        start_dt = row["datetime"]
                        snow_acc.append(row["p1h"])
                        temp.append(row["temp"])

                else:
                    snow_acc.append(row["p1h"])
                    temp.append(row["temp"])
            else:
                if detected_storm:
                    stop_dt = row["datetime"]
                    duration = stop_dt - start_dt

                    new_row = pd.DataFrame([{"Start Datetime": start_dt, 
                                             "End Datetime" : stop_dt , 
                                             "Duration" : duration, 
                                             "Precipitation by Hour" : snow_acc, 
                                             "Temp by Hour" : temp
                                             }])
                    if sum(snow_acc) > 4: #only log as storm if there is >4mm of snow
                        storm_df = pd.concat([storm_df, new_row], ignore_index=True)

                    snow_acc = []
                    temp = []
                    detected_storm = False

        if detected_storm:
                    stop_dt = row["datetime"]
                    duration = stop_dt - start_dt
                    new_row = pd.DataFrame([{"Start Datetime": start_dt, 
                                             "End Datetime" : stop_dt , 
                                             "Duration" : duration, 
                                             "Precipitation by Hour" : snow_acc, 
                                             "Temp by Hour" : temp
                                             }])
                    if sum(snow_acc) > 4: #only log as storm if there is >4mm of snow
                        storm_df = pd.concat([storm_df, new_row], ignore_index=True)

        storm_df.to_csv(self.storm_csv_path, index=False)


    @staticmethod
    def fetch_data(station = "Central Park", start_year = 2010, end_year = 2025):

        base_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"
        years = range(start_year, end_year)

        print(f"Fetching NYC weather info from {base_url} for years {start_year}-{end_year}")

        raw_dir = REPO_ROOT / "data" /"raw" / "isd_lite"
        processed_dir = REPO_ROOT / "data" / "processed"

        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        csv_out = processed_dir / "isd_nyc.csv"

        if type(station) == str:
            station = [station]

        stations_to_process = []
        for s in station:
            if s == "Central Park":
                stations_to_process.append(("725053", "94728"))
            elif s == "LaGuardia":
                stations_to_process.append(("725030", "14732"))
            elif s == "JFK":
                stations_to_process.append(("744860", "94789"))
            else:
                raise ValueError(f"Station {s} is not in NYC, please use only 'Central Park', 'LaGuardia' or 'JFK'")


        frames = []
        for usaf, wban in stations_to_process:
            for year in years:
                fname = f"{usaf}-{wban}-{year}.gz"
                save_path = raw_dir / fname
                url = f"{base_url}/{year}/{fname}"
                if save_path.exists():
                    pass
                else:
                    try:
                        urllib.request.urlretrieve(url, save_path)
                        print(f"Downloaded {fname}")
                    except Exception as e:
                        print(e)
                        continue

                df = pd.read_csv(
                    save_path,
                    sep=r"\s+",
                    names=["year", "month", "day", "hour","temp", "dew", "slp", "wdir", "wspd", "skc", "p1h", "p6h"],
                    na_values=[-9999],
                    compression="gzip",
                    engine="python",
                )

                df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]], utc=True)
                df["usaf"] = usaf
                df["wban"] = wban

                for col in ["temp", "dew", "slp", "wspd", "p1h", "p6h"]:
                    df[col] = df[col] / 10.0

                frames.append(df)

        df = pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
        df.to_csv(csv_out, index=False)

        print(f"Saved Processed .csv at {csv_out}")

        







            




        
    




        
        
