import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor




env = gym.make('Pendulum-v1')
env = Monitor(env,'ppo_continuos_pendulam')


env = DummyVecEnv([lambda: env])


clip_range =  0.2
ent_coef = 0.0
gae_lambda = 0.95
gamma = 0.9

learning_rate = 0.001
n_envs = 4
n_epochs = 10
n_steps = 1024

n_timesteps = 1000000
policy='MlpPolicy'
sde_sample_freq = 4
use_sde = True



model = PPO(verbose=1,tensorboard_log='ppocontinuos/pendulum',policy=policy,env=env, clip_range=clip_range, ent_coef=ent_coef,gae_lambda=gae_lambda,gamma=gamma,learning_rate=learning_rate,n_steps=n_steps,n_epochs=n_epochs,sde_sample_freq=sde_sample_freq,use_sde=use_sde)


model.learn(total_timesteps=n_timesteps,progress_bar=True)

model.save("ppo_Pendulum")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
env.close()