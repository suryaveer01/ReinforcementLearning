import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import torch
from stable_baselines3.common.monitor import Monitor
import os





# Load the saved statistics
vec_env = DummyVecEnv([lambda: gym.make("Walker2d-v4",render_mode ='human')])
vec_env = VecNormalize.load("Monitor_logs/vecnormalize.pkl", vec_env)
#  do not update them at test time
vec_env.training = False
# reward normalization is not needed at test time
vec_env.norm_reward = False

# Load the agent
model = PPO.load("walker_2d_ppo_tuned", env=vec_env,print_system_info=True)



mean_reward, std_reward = evaluate_policy(model, vec_env,render=True, n_eval_episodes=5)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
vec_env.close()