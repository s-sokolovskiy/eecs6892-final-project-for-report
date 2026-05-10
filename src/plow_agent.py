import numpy as np
import tensorflow as tf
from src.env import WinterStorm
from src.road_network import RoadNetwork
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from shapely.geometry import LineString
from matplotlib.lines import Line2D
from pathlib import Path
import pickle
import time
from src.dnns import GNEncoder, Actor, Critic


REPO_ROOT = Path(__file__).parent.parent

class PlowAgent:

    def __init__(self, num_plows = 50, tick_size = 1):

        self.road_network = RoadNetwork()
        self.tick_size = tick_size
        self.num_plows = num_plows

    def train_agent(self, gamma = 0.95, last_n_reward = 100, train_episodes = 50, actor_lr = 1e-4, critic_lr = 1e-4, ppo_epochs = 5, load_saved_weights = False):
        self.gamma = gamma
        
        self.actor = self.create_node_scoring_network()
        self.critic = self.create_critic_network()
        self.encoder = self.create_graph_encoder_layer()

        if load_saved_weights:
            self.actor(tf.zeros((1, self.road_network.max_degree, self.num_actor_features)))
            self.critic(tf.zeros((1, 16)))
            self.encoder(tf.zeros((1, self.road_network.N, 5)),
                        tf.zeros((1, self.road_network.E, 5)),
                        self.road_network.node_idxs_on_edges,
                        self.road_network.N
                        )


            save_path = REPO_ROOT / 'data' / 'models'
            self.actor.load_weights(save_path / 'actor.weights.h5')
            self.critic.load_weights(save_path / 'critic.weights.h5')
            self.encoder.load_weights(save_path / 'encoder.weights.h5')

        # test_state = np.random.randn(self.road_network.E  * 2 + self.num_plows * 5).astype(np.float32)
        # test_out = self.actor(tf.convert_to_tensor(test_state[None, :]))
        # print("Initial actor output:", test_out.numpy(), "has nan:", np.isnan(test_out.numpy()).any())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr, clipnorm = 0.5)  
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr, clipnorm = 0.5)

        self.ppo_epochs = ppo_epochs

        # self.episode_reward_history = [] #total reward for each episode (total amount of snow cleared in episode)
        self.avg_rewards_per_tick = [] #average reward per tick for each episode (average amount of snow cleared per tick in episode)
        self.running_avg_rewards_per_tick = []
        self.total_rewards = [] #total reward collected in an episode

        #dicts that store stuff for training actor/critic networks         
        snow_accumulations = {}
        traffic_demands = {}
        rewards = {}
        decisions = {}

        train_pool_size = 5

        for episode in tqdm(range(train_episodes)):

            decisions[episode],  rewards[episode], snow_accumulations[episode], traffic_demands[episode] = self.sample_episode()
            self.avg_rewards_per_tick.append(np.mean(rewards[episode]))
            self.total_rewards.append(np.sum(rewards[episode]))

            if episode % train_pool_size == 0:

                self.optim(decisions, snow_accumulations, traffic_demands, rewards)
                snow_accumulations = {}
                traffic_demands = {}
                rewards = {}
                decisions = {}

                print("Updated Policy")
            

            if len(self.avg_rewards_per_tick) > last_n_reward:
                running_avg_rewards_per_tick_ep = np.mean(self.avg_rewards_per_tick[-last_n_reward:]) 
            else:
                running_avg_rewards_per_tick_ep = np.mean(self.avg_rewards_per_tick) 
            self.running_avg_rewards_per_tick.append(running_avg_rewards_per_tick_ep)

            print(f"Episode: {episode + 1}, Avg Reward per Tick: {self.avg_rewards_per_tick[-1]}, Running Avg Reward per Tick: {running_avg_rewards_per_tick_ep}")
        

        save_path = REPO_ROOT / 'data' / 'models'
        save_path.mkdir(parents=True, exist_ok=True)
        self.actor.save_weights(save_path / f'actor_{time.strftime("%Y-%m-%d_%H-%M")}.weights.h5')
        self.critic.save_weights(save_path / f'critic_{time.strftime("%Y-%m-%d_%H-%M")}.weights.h5')
        self.encoder.save_weights(save_path / f'encoder_{time.strftime("%Y-%m-%d_%H-%M")}.weights.h5')

        with open(save_path / f'reward_history_{time.strftime("%Y-%m-%d_%H-%M")}.pkl', 'wb') as f:
            pickle.dump({
                'avg_rewards_per_tick': self.avg_rewards_per_tick,
                'running_avg_rewards_per_tick': self.running_avg_rewards_per_tick,
                'total_rewards' : self.total_rewards
            }, f)

        fig, ax = plt.subplots()
        ax.plot(self.avg_rewards_per_tick, label = 'episode reward')
        ax.set_xlabel("Episode #")
        ax.set_ylabel("Average Amount of Snow Cleared Per Tick")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        ax.set_title("PLOWR Average Reward (Amount of Snow Cleared) per Tick History")
        save_path = REPO_ROOT / 'data' / 'figs'
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f'avg_reward_per_tick_hist_{time.strftime("%Y-%m-%d_%H-%M")}.pdf')
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(self.running_avg_rewards_per_tick, label = 'episode reward')
        ax.set_xlabel("Episode #")
        ax.set_ylabel(f"Running Average Amount of Snow Cleared Per Tick across last {last_n_reward} episodes")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        ax.set_title(f"PLOWR Running Average Amount of Snow Cleared Per Tick across last {last_n_reward} episodes")
        save_path = REPO_ROOT / 'data' / 'figs'
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f'running_avg_reward_per_tick_hist_{time.strftime("%Y-%m-%d_%H-%M")}.pdf')
        plt.close()


    def test_agent(self, test_episodes  = 100):

        self.actor = self.create_node_scoring_network()
        self.critic = self.create_critic_network()
        self.actor.load_weights(save_path / 'actor.weights.h5')
        self.critic.load_weights(save_path / 'critic.weights.h5')

        test_episode_reward_history = [] 
        test_episode_reward_history_norm = []
        snow_accumulations = {}
        traffic_demands = {}
        rewards = {}
        decisions = {}


        for episode in tqdm(range(test_episodes)):

            decisions[episode],  rewards[episode], snow_accumulations[episode], traffic_demands[episode], reward_hist_normalized = self.sample_episode()
            test_episode_reward_history.append(np.mean(rewards[episode]))
            test_episode_reward_history_norm.append(np.mean(reward_hist_normalized))


            print(f"Episode: {episode + 1}, Episode Reward: {test_episode_reward_history_norm[-1]}")
        
        fig, ax = plt.subplots()
        ax.plot(test_episode_reward_history_norm)
        ax.set_xlabel("Episode #")
        ax.set_ylabel("Episode Reward")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        ax.set_title("PLOWR Normalized Episode Test Reward History")
        save_path = REPO_ROOT / 'data' / 'figs'
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / "test_reward_hist.pdf")
        plt.close()


    def create_graph_encoder_layer(self, num_layers = 10):

        edge_in_dim = 5
        node_in_dim = 5
        edge_out_dim = 5
        node_out_dim =  5

        return GNEncoder(num_layers, edge_in_dim, node_in_dim, edge_out_dim, node_out_dim, self.road_network.M)


    def create_critic_network(self):
        return Critic()

    def create_node_scoring_network(self):
        self.num_actor_features = 5 + 5 + 16
        return Actor(max_num_edges = self.road_network.max_degree, num_features = self.num_actor_features)
    
    def update_plows(self, snow_accumulation, traffic_demand, plows, current_tick,  decisions, snow_accumulations, traffic_demands):

        for plow in plows:
            if np.abs(plow[3] - 1.0) <= 1e-6:
                next_options = self.road_network.next_options[plow[1]]

                if len(next_options) == 0:
                    # dead end: teleport to a random valid edge
                    random_idx = np.random.choice(self.road_network.E)
                    u, v, key = self.road_network.edges[random_idx]
                    plow[0] = self.road_network.node_to_idx[u]
                    plow[1] = self.road_network.node_to_idx[v]
                    plow[2] = random_idx
                    plow[3] = 0.0
                    continue

                next_option_scores = np.zeros(len(next_options))

                plows_heading_to = np.zeros(self.road_network.N)
                plows_leaving_from = np.zeros(self.road_network.N)

                for p in plows:
                    plows_heading_to[int(p[1])] += 1
                    plows_leaving_from[int(p[0])] +=1

                node_features = np.column_stack([self.road_network.node_coords, self.road_network.node_out_degree, plows_heading_to, plows_leaving_from]) #matrix of shape (N, 5)
                edge_features = np.column_stack([snow_accumulation, traffic_demand, self.road_network.length, self.road_network.max_speed, self.road_network.num_lanes])  #stack arrays of len (E,)|(E,)|(E,)|(E,)|(E,)

                node_features, edge_features, global_representation = self.encoder(node_features[None,:], edge_features[None,:],  self.road_network.node_idxs_on_edges, self.road_network.N)                          

                batched_feature_for_node = []
                for i, option in enumerate(next_options):
                    batched_feature_for_node.append(tf.concat([node_features[0, option[0]], edge_features[0,option[1]], tf.squeeze(global_representation)], axis = 0)[None, :])
                batched_feature_for_node = tf.concat(batched_feature_for_node, axis =0)

                next_option_scores = self.actor(batched_feature_for_node)  
                    
                plows_to_save = plows.copy()
                plow[0] = plow[1]
                p = next_option_scores.numpy().astype(np.float64)
                p = p / p.sum()
                next_node_idx =  np.random.choice(range(len(next_option_scores)), p=p)
                plow[1] = next_options[next_node_idx][0]
                plow[2] = next_options[next_node_idx][1]
                plow[3] = 0.0
                plow[4] = 1 #always plowing for simplicity

                eps = 1e-9
                log_p_node = tf.math.log(next_option_scores[next_node_idx] + eps)

                decisions.append([plows_to_save, next_options, next_node_idx, plow[4], current_tick, log_p_node])
                snow_accumulations[current_tick] = snow_accumulation.copy()
                traffic_demands[current_tick] = traffic_demand.copy()

        return plows, decisions, snow_accumulations, traffic_demands
    

    def sample_episode(self):

        decisions = []
        snow_accumulations = {}
        traffic_demands = {}

        reward_hist = []
        # reward_hist_normalized = []
        snow_accumulation = np.zeros(self.road_network.E)
        traffic_demand = np.zeros(self.road_network.E)
        reward = 0
        done = False

        start_hour = np.random.randint(168)

        plows = np.zeros(shape = (self.num_plows, 5), dtype = np.float32)

        plows[:,2] = np.random.choice(self.road_network.E, size=self.num_plows)
        for i in range(self.num_plows):
            u, v, key = self.road_network.edges[int(plows[i,2])]
            plows[i,0] = self.road_network.node_to_idx[u]
            plows[i,1] = self.road_network.node_to_idx[v]
            plows[i,4] = 1
        # assign random initial positions
        #assume each plow stores (start node, end node, edge idx, position, mode: 1.0 for plowing and 0.0 for driving)
        
        env = WinterStorm(start_hour, self.road_network, self.tick_size)
        # total_snow_fallen = env.snow_model.precipitation_per_step.sum()        
        # storm_length = np.shape(env.snow_model.precipitation_per_step)[1]
        while not done:

            # counter += 1
            # print(counter)
            snow_accumulation, traffic_demand, plows, reward, done = env.step(plows)
            plows, decisions, snow_accumulations, traffic_demands = self.update_plows(snow_accumulation, traffic_demand, plows, env.current_tick, decisions, snow_accumulations, traffic_demands)
            reward_hist.append(reward)

            # normalized_reward = reward / (total_snow_fallen + 1e-9)
            # reward_hist_normalized.append(normalized_reward)

        return decisions, reward_hist, snow_accumulations, traffic_demands, #storm_length
    


    def optim(self, decisions, snow_accumulations, traffic_demands, rewards, batch_size = 256):

        episodes = decisions.keys()
        all_node_feats = []
        all_edge_feats = []
        all_next_options = []
        all_chosen_idxs = []
        all_log_p_old = []
        all_G_t = []
    

        for episode in episodes:

            returns_by_tick = compute_discounted_reward(rewards[episode], self.gamma)

            for i in range(len(decisions[episode])):

                if decisions[episode][i][4] >= len(returns_by_tick):
                    continue

                plows, next_options, next_node_idx, next_mode, current_tick, log_p_action = decisions[episode][i]
                snow_accumulation = snow_accumulations[episode][current_tick]
                traffic_demand = traffic_demands[episode][current_tick]

                plows_heading_to = np.zeros(self.road_network.N)
                plows_leaving_from = np.zeros(self.road_network.N)

                for p in plows:
                    plows_heading_to[int(p[1])] += 1
                    plows_leaving_from[int(p[0])] +=1

                node_features = np.column_stack([self.road_network.node_coords, self.road_network.node_out_degree, plows_heading_to, plows_leaving_from]).astype(np.float32) #matrix of shape (N, 5)
                edge_features = np.column_stack([snow_accumulation, traffic_demand, self.road_network.length, self.road_network.max_speed, self.road_network.num_lanes]).astype(np.float32)  #stack arrays of len (E,)|(E,)|(E,)|(E,)|(E,)


                all_edge_feats.append(edge_features)
                all_node_feats.append(node_features)
                all_next_options.append(next_options)
                all_chosen_idxs.append(next_node_idx)
                all_log_p_old.append(log_p_action)
                all_G_t.append(returns_by_tick[decisions[episode][i][4]])


        all_node_feats_initial = tf.stack(all_node_feats)
        all_edge_feats_initial = tf.stack(all_edge_feats)
        # all_next_options = #need to convert to TF tensor (maybe ?)
        all_next_options = all_next_options
        all_chosen_idxs = tf.stack(all_chosen_idxs, axis=0)
        all_log_p_old = tf.convert_to_tensor(all_log_p_old)
        all_G_t = tf.convert_to_tensor(all_G_t, dtype=tf.float32)

    
        for _ in range(self.ppo_epochs):
            
            D = len(all_G_t)
            perm = np.random.permutation(range(D))

            for start in range(0, D, batch_size):

                batch_idxs = perm[start : start + batch_size]

                with tf.GradientTape() as tape:

                    all_node_feats_initial_minibatch = tf.gather(all_node_feats_initial,batch_idxs)
                    all_edge_feats_initial_minibatch = tf.gather(all_edge_feats_initial,batch_idxs)
                    all_next_options_minibatch = [all_next_options[i] for i in batch_idxs]
                    all_chosen_idxs_minibatch = tf.gather(all_chosen_idxs,batch_idxs)
                    all_log_p_old_minibatch = tf.gather(all_log_p_old,batch_idxs)
                    all_G_t_minibatch = tf.gather(all_G_t,batch_idxs)


                    all_node_feats, all_edge_feats, all_global = self.encoder(all_node_feats_initial_minibatch, all_edge_feats_initial_minibatch,  self.road_network.node_idxs_on_edges, self.road_network.N)                          

                    G_t_pred = tf.squeeze(self.critic(all_global),axis = -1)
                    targets = tf.convert_to_tensor(all_G_t_minibatch, dtype = tf.float32)
                    critic_loss = tf.reduce_mean((targets - G_t_pred) ** 2)
    
                    advantages = tf.convert_to_tensor(all_G_t_minibatch) - tf.stop_gradient(G_t_pred)
                    advantages = (advantages - tf.math.reduce_mean(advantages, axis = 0)) / (tf.math.reduce_std(advantages, axis = 0) + 1e-9)

                    # all_actor_inputs = []
                    # for j in range(len(all_chosen_idxs)):
                    #     batched_feature_for_node = []
                    #     for i, option in enumerate(all_next_options[j]):
                    #         batched_feature_for_node.append(tf.concat([all_node_feats[j, option[0]], all_edge_feats[j, option[1]], tf.squeeze(all_global[j])])[None, :])
                    #     all_actor_inputs.append(tf.concat(batched_feature_for_node))
                    # all_actor_inputs = tf.concat(all_actor_inputs)

                    all_actor_inputs = []
                    for j, options in enumerate(all_next_options_minibatch):
                        cand_features = []
                        for option in options:
                            feat = tf.concat([all_node_feats[j, option[0]], all_edge_feats[j, option[1]], all_global[j]], axis=0)
                            cand_features.append(feat)
                        while len(cand_features) < self.road_network.max_degree:
                            cand_features.append(tf.zeros((self.num_actor_features,), dtype=tf.float32))
                        all_actor_inputs.append(tf.stack(cand_features, axis=0))
                    all_actor_inputs = tf.stack(all_actor_inputs, axis=0)


                    actor_out = self.actor(all_actor_inputs)
                    chosen_node_probs = tf.gather(actor_out, all_chosen_idxs_minibatch, batch_dims = 1)

                    eps = 1e-9
                    all_log_p_new = tf.math.log(chosen_node_probs + eps) 
                    ratio = tf.exp(all_log_p_new - all_log_p_old_minibatch)

                    ratio_eps = 0.2
                    surrogate = tf.minimum(ratio * advantages, tf.clip_by_value(ratio , 1 - ratio_eps, 1 + ratio_eps) * advantages)
                    actor_loss = -tf.reduce_mean(surrogate)
                    entropy = -tf.reduce_sum(actor_out * tf.math.log(actor_out + eps), axis=-1)
                    actor_loss = actor_loss - 0.01 * tf.reduce_mean(entropy)

                    loss = actor_loss + critic_loss
                    

                actor_enc_vars = self.actor.trainable_variables + self.encoder.trainable_variables
                critic_vars = self.critic.trainable_variables
                grads = tape.gradient(loss, actor_enc_vars + critic_vars)
                n = len(actor_enc_vars)
                self.actor_optimizer.apply_gradients(zip(grads[:n], actor_enc_vars))
                self.critic_optimizer.apply_gradients(zip(grads[n:], critic_vars))


    def _ensure_actor(self): #here actor can just load incorrect/trained weights if you actually want random for sanity check
        if not hasattr(self, 'actor') or self.actor is None:
            self.actor = self.create_node_scoring_network()
            self.encoder = self.create_graph_encoder_layer()
            try:
                self.actor.load_weights(REPO_ROOT / 'data' / 'models' / 'actor.weights.h5')
                self.actor.load_weights(REPO_ROOT / 'data' / 'models' / 'encoder.weights.h5')
            except:
                print("Using Random Wights for the Actor")


    def _collect_episode_states(self, seed=None):
        self._ensure_actor()

        snow_hist = []
        traffic_hist = []
        plow_hist = []

        snow_accumulation = np.zeros(self.road_network.E)
        traffic_demand = np.zeros(self.road_network.E)
        done = False

        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(seed)

        start_hour = np.random.randint(168)
        env = WinterStorm(start_hour, self.road_network, self.tick_size)

        plows = np.zeros(shape=(self.num_plows, 5), dtype=np.float32)
        plows[:, 2] = np.random.choice(self.road_network.E, size=self.num_plows)
        for i in range(self.num_plows):
            u, v, key = self.road_network.edges[int(plows[i, 2])]
            plows[i, 0] = self.road_network.node_to_idx[u]
            plows[i, 1] = self.road_network.node_to_idx[v]
            plows[i, 4] = 1

        if seed is not None:
            np.random.set_state(rng_state)

        while not done:
            snow_accumulation, traffic_demand, plows, _, done = env.step(plows)
            plows, _, _, _ = self.update_plows(snow_accumulation, traffic_demand, plows, env.current_tick, [], {}, {})
            snow_hist.append(snow_accumulation.copy())
            traffic_hist.append(traffic_demand.copy())
            plow_hist.append(plows.copy())

        return snow_hist, traffic_hist, plow_hist, start_hour


    def _edge_centerlines(self):
        if hasattr(self, '_centerlines_cache'):
            return self._centerlines_cache

        G = self.road_network.G
        centerlines = []

        for (u, v, k) in self.road_network.edges:
            data = G.edges[u, v, k]
            if 'geometry' in data:
                geom = data['geometry']
            else:
                x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
                x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
                geom = LineString([(x1, y1), (x2, y2)])
            centerlines.append(geom)

        self._centerlines_cache = centerlines
        return centerlines


    def plot_state(self, tick, snow_hist, traffic_hist, plow_hist, start_hour,
                   snow_max=None, traffic_max=None, axes=None, add_colorbar=True):
        centerlines = self._edge_centerlines()
        base_lines = [list(g.coords) for g in centerlines]

        snow = snow_hist[tick]
        traffic = traffic_hist[tick]
        plows = plow_hist[tick]

        if snow_max is None:
            snow_max = max(float(snow.max()), 1e-9)
        if traffic_max is None:
            traffic_max = max(float(traffic.max()), 1e-9)

        snow_norm = mcolors.Normalize(vmin=0, vmax=snow_max)
        traffic_norm = mcolors.Normalize(vmin=0, vmax=traffic_max)

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        else:
            fig = axes[0].figure
        ax_snow, ax_traffic = axes

        plowing_xy = []
        driving_xy = []
        for plow in plows:
            edge_idx = int(plow[2])
            frac = float(np.clip(plow[3], 0.0, 1.0))
            pt = centerlines[edge_idx].interpolate(frac, normalized=True)
            if int(plow[4]) == 1:
                plowing_xy.append((pt.x, pt.y))
            else:
                driving_xy.append((pt.x, pt.y))

        legend_handles = [
            Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=8, label='plowing'),
            Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=8, markerfacecolor='none', label='driving'),
        ]

        for ax, values, norm, cmap in [
            (ax_snow, snow, snow_norm, cm.Blues),
            (ax_traffic, traffic, traffic_norm, cm.Reds),
        ]:
            ax.set_aspect('equal')
            ax.set_facecolor('white')
            ax.add_collection(LineCollection(base_lines, colors='lightgray', linewidths=0.3, zorder=1))
            ax.add_collection(LineCollection(base_lines, colors=cmap(norm(values)), linewidths=1.5, zorder=2))

            if plowing_xy:
                xs, ys = zip(*plowing_xy)
                ax.scatter(xs, ys, marker='x', c='black', s=30, zorder=5)
            if driving_xy:
                xs, ys = zip(*driving_xy)
                ax.scatter(xs, ys, marker='o', s=25, zorder=5, facecolors='none', edgecolors='black')

            ax.autoscale_view()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(handles=legend_handles, loc='upper right')

        if add_colorbar:
            fig.colorbar(cm.ScalarMappable(norm=snow_norm, cmap='Blues'), ax=ax_snow, label='snow level', fraction=0.04, pad=0.02)
            fig.colorbar(cm.ScalarMappable(norm=traffic_norm, cmap='Reds'), ax=ax_traffic, label='traffic level', fraction=0.04, pad=0.02)

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        total_minutes = (start_hour * 60 + tick * self.tick_size) % (7 * 24 * 60)
        day_idx = total_minutes // (24 * 60)
        time_of_day = total_minutes % (24 * 60)
        hh = time_of_day // 60
        mm = time_of_day % 60
        timestamp = f"{day_names[day_idx]} {hh:02d}:{mm:02d}"
        ax_snow.set_title(f"snow — {timestamp}")
        ax_traffic.set_title(f"traffic — {timestamp}")

        return fig, axes


    def animate_episode(self, save_path, tick_step=1, fps=8, seed=None):
        save_path = Path(save_path)
        snow_hist, traffic_hist, plow_hist, start_hour = self._collect_episode_states(seed=seed)

        snow_max = max((float(arr.max()) for arr in snow_hist), default=1e-9)
        traffic_max = max((float(arr.max()) for arr in traffic_hist), default=1e-9)

        T = len(snow_hist)
        frames = list(range(0, T, tick_step))

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        ax_snow, ax_traffic = axes

        snow_norm = mcolors.Normalize(vmin=0, vmax=snow_max)
        traffic_norm = mcolors.Normalize(vmin=0, vmax=traffic_max)
        fig.colorbar(cm.ScalarMappable(norm=snow_norm, cmap='Blues'), ax=ax_snow, label='snow level', fraction=0.04, pad=0.02)
        fig.colorbar(cm.ScalarMappable(norm=traffic_norm, cmap='Reds'), ax=ax_traffic, label='traffic level', fraction=0.04, pad=0.02)

        def update(frame_tick):
            for ax in axes:
                ax.clear()
            self.plot_state(frame_tick, snow_hist, traffic_hist, plow_hist, start_hour,
                            snow_max=snow_max, traffic_max=traffic_max, axes=axes, add_colorbar=False)
            return []

        anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False)

        suffix = save_path.suffix.lower()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == '.mp4':
            writer = animation.FFMpegWriter(fps=fps)
        elif suffix == '.gif':
            writer = animation.PillowWriter(fps=fps)
        else:
            raise ValueError(f"Unsupported animation format {suffix}; use .mp4 or .gif")

        anim.save(save_path, writer=writer)
        plt.close(fig)


def compute_discounted_reward(rewards, gamma):

    G_t = np.zeros(len(rewards))
    G_t[-1] = rewards[-1]
    for t in range(len(rewards) - 2, -1, -1):
        G_t[t] = rewards[t] + gamma * G_t[t + 1]

    return G_t

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)