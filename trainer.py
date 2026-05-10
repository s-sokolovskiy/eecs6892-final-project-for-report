import os 
import tensorflow as tf
import warnings 
import time
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={os.environ["CONDA_PREFIX"]}'
print(tf.config.list_physical_devices())


from src.plow_agent import PlowAgent


plow_agent = PlowAgent(num_plows= 10, tick_size = 1)
plow_agent.animate_episode(f'data/figs/before_training_{time.strftime("%Y-%m-%d_%H-%M")}.mp4', tick_step=1, fps=1, seed = 777)
plow_agent.train_agent(train_episodes=200, load_saved_weights=False)
plow_agent.animate_episode(f'data/figs/after_training_{time.strftime("%Y-%m-%d_%H-%M")}.mp4', tick_step=1, fps=1, seed=777)
