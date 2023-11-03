import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results
import matplotlib.pyplot as plt





env = gym.make('CliffWalking-v0')

# env = DummyVecEnv([lambda: env])

env = Monitor(env,'part1_discrete_cliffwalking_train')

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                                  net_arch=[64, 32])

lr = 0.0003
buffer_size = 10_000
learning_starts = 5_000 
batch_size = 128 
gamma = 0.97 
target_update = 100 
exploration_fraction = 0.15
final_eps = 0.005



model = DQN('MlpPolicy',env,
            target_update_interval=target_update,
              exploration_fraction = exploration_fraction,exploration_final_eps=final_eps,
              learning_rate=lr,policy_kwargs=policy_kwargs,
              buffer_size=buffer_size,learning_starts=learning_starts,
              batch_size=batch_size,gamma=gamma,
              tensorboard_log='dqndiscrete/cliffwalking',
                verbose=1)
model.learn(total_timesteps=200000)

model.save("dqn_cliffwalking")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
env.close()