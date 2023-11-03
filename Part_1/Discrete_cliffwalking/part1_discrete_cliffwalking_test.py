from stable_baselines3 import DQN
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results
import matplotlib.pyplot as plt


# Load the trained model


# Create the Taxi-v4 environment
env = gym.make('CliffWalking-v0',render_mode = 'human')


loaded_model = DQN.load("dqn_cliffwalking_tuned",env=env,device='cpu', print_system_info=True)

print(loaded_model)

# Test the loaded model for a specified number of episodes
# mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


mean_reward, std_reward = evaluate_policy(loaded_model, env,render=True, n_eval_episodes=1)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


df = load_results('./Monitor_logs/')

plt.plot(df['r'])
plt.ylabel('Reward')
plt.xlabel('Steps')
plt.show()

# plt.close()

plt.plot(df['l'].cumsum(), df['r'].rolling(window=10).mean())
plt.ylabel('Average Reward')
plt.xlabel('Steps')
plt.show()