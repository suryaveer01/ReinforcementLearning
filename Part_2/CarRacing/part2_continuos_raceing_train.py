import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from torch import nn



# Create the LunarLander-v2 environment
env = gym.make('CarRacing-v2')

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

n_envs= 8  # You can adjust the number of parallel environments
env = SubprocVecEnv([make_env('CarRacing-v2', i + 32) for i in range(n_envs)],start_method='fork')

frame_stack= 2
normalize= "{'norm_obs': False, 'norm_reward': True}"

n_timesteps= 4e6
policy= 'CnnPolicy'
batch_size= 128
n_steps= 512
gamma= 0.99
gae_lambda= 0.95
n_epochs= 10
ent_coef= 0.0
sde_sample_freq= 4
max_grad_norm= 0.5
vf_coef= 0.5
learning_rate= 1e-4
use_sde= True
clip_range= 0.2
policy_kwargs= dict(log_std_init=-2,ortho_init=False, activation_fn=nn.GELU, net_arch=dict(pi=[256], vf=[256]))

# Define and create the SAC agent
model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs,
             verbose=1,tensorboard_log='ppocontinuos/',
             batch_size=batch_size, n_steps=n_steps, gamma=gamma, gae_lambda=gae_lambda,
             ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
             learning_rate=learning_rate, use_sde=use_sde, clip_range=clip_range,
             sde_sample_freq=sde_sample_freq, n_epochs=n_epochs,
             )

# Train the agent
model.learn(total_timesteps=n_timesteps)

# Save the trained model
model.save("ppo_car_racing")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
env.close()