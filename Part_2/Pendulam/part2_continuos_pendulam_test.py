import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results
import matplotlib.pyplot as plt



env = gym.make('Pendulum-v1',render_mode = 'human')

env = DummyVecEnv([lambda: env])
model = PPO.load("ppo_Pendulum_tuned",device='cpu')



# model = PPO(verbose=1,tensorboard_log='ppocontinuos/pendulum',policy=policy,env=env, clip_range=clip_range, ent_coef=ent_coef,gae_lambda=gae_lambda,gamma=gamma,learning_rate=learning_rate,n_steps=n_steps,n_epochs=n_epochs,sde_sample_freq=sde_sample_freq,use_sde=use_sde)


# model.learn(total_timesteps=50000)

# model.save("ppo_Pendulum")

mean_reward, std_reward = evaluate_policy(model, env,render=True, n_eval_episodes=5)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
env.close()


df = load_results('./logs/')

plt.plot(df['r'])
plt.ylabel('Reward')
plt.xlabel('Steps')
plt.show()

# plt.close()

plt.plot(df['l'].cumsum(), df['r'].rolling(window=10).mean())
plt.ylabel('Average Reward')
plt.xlabel('Steps')
plt.show()