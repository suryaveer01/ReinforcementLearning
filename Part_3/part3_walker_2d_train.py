import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import torch
from stable_baselines3.common.monitor import Monitor
import os




# env = gym.make("Walker2d-v4")
vec_env = DummyVecEnv([lambda: gym.make("Walker2d-v4")])

# vec_env = Monitor(vec_env,'walker2d_mujoco')

# env = gym.wrappers.RecordEpisosdeStatistics(env)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)

# env = DummyVecEnv([lambda: env])


n_timesteps= 1e6
batch_size= 32
n_steps= 512
gamma= 0.99
learning_rate= 5.05041e-05
ent_coef= 0.000585045
clip_range= 0.1
n_epochs= 20
gae_lambda= 0.95
max_grad_norm= 1
vf_coef= 0.871923


# Train the agent

model = PPO("MlpPolicy", vec_env,batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,learning_rate=learning_rate,
            ent_coef=ent_coef,clip_range=clip_range,
            n_epochs=n_epochs,gae_lambda=gae_lambda
            ,max_grad_norm=max_grad_norm,vf_coef=vf_coef,
                verbose=1, tensorboard_log="a2cWalker2d")

model.learn(total_timesteps=n_timesteps,progress_bar=True)


log_dir = "/walker2d/"
# model.save("/walker2d/ppo_walker2d")
model.save("walker_2d_ppo")
vec_env.save("vecnormalize.pkl")

# To demonstrate loading
del model, vec_env

# Load the saved statistics
vec_env = DummyVecEnv([lambda: gym.make("Walker2d-v4")])
vec_env = VecNormalize.load("vecnormalize.pkl", vec_env)
#  do not update them at test time
vec_env.training = False
# reward normalization is not needed at test time
vec_env.norm_reward = False

# Load the agent
model = PPO.load("walker_2d_ppo", env=vec_env)



mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=5)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
vec_env.close()