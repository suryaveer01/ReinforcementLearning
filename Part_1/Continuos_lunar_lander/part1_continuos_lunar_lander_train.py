import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


# Create the LunarLander-v2 environment
env = gym.make('LunarLander-v2')

env = Monitor(env,'dqn_continuos_lunar_lander')

policy_args = dict(net_arch = [256,256])
# Define and create the DQN agent
model = DQN('MlpPolicy',learning_rate=0.00063,batch_size=128,
            buffer_size=50000,exploration_final_eps=0.1,exploration_fraction=0.12,
            gamma=0.99,gradient_steps=-1,learning_starts=0, target_update_interval=250,
            train_freq=4,policy_kwargs=policy_args,env=env,tensorboard_log="dqn_continuos/",  verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the trained model
model.save("dqn_lunarlander")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
env.close()