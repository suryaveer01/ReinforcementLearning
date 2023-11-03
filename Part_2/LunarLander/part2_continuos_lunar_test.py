import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv




env = gym.make('LunarLander-v2',continuous = True,render_mode = 'human')


observation_space = env.observation_space
action_space = env.action_space

custom_objects={'action_space': action_space,'observation_space': observation_space}
env = DummyVecEnv([lambda: env])
model = SAC.load("sac_lunarlander_cont_tuned",device='cpu',custom_objects = custom_objects)



# model = PPO(verbose=1,tensorboard_log='ppocontinuos/pendulum',policy=policy,env=env, clip_range=clip_range, ent_coef=ent_coef,gae_lambda=gae_lambda,gamma=gamma,learning_rate=learning_rate,n_steps=n_steps,n_epochs=n_epochs,sde_sample_freq=sde_sample_freq,use_sde=use_sde)


# model.learn(total_timesteps=50000)

# model.save("ppo_Pendulum")

mean_reward, std_reward = evaluate_policy(model, env,render=True, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
env.close()