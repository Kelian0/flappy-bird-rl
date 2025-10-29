import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt 

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os

DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self, hyperparameters_set="carpole1"):
        with open("hyperparameters.yml", "r") as file:
            all_hyperparameters_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_set[hyperparameters_set]

        self.hyperparameters_set = hyperparameters_set

        self.env_id             = hyperparameters["env_id"]
        self.learning_rate_a    = hyperparameters["learning_rate_a"]
        self.discount_factor_g  = hyperparameters["discount_factor_g"]
        self.network_sync_rate  = hyperparameters["network_sync_rate"]
        self.replay_memory_size = hyperparameters["replay_memory_size"]     # size of the replay memory
        self.mini_batch_size    = hyperparameters["mini_batch_size"]        # size of the mini-batch
        self.epsilon_init       = hyperparameters["espilon_init"]           # initial value of epsilon 1 = 100% random actions
        self.epsilon_decay      = hyperparameters["espilon_decay"]          # decay rate of epsilon
        self.epsilon_min        = hyperparameters["espilon_min"]            # minimum value of epsilon
        self.stop_on_reward     = hyperparameters["stop_on_reward"]
        self.fc1_nodes          = hyperparameters["fc1_nodes"]
        self.env_make_params    = hyperparameters.get("env_make_params", {}) 

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.png')

    def run(self, is_training=True, render=False):

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Starting training... \n"
            print(log_message)
            with open(self.LOG_FILE, "a") as file:
                file.write(log_message)

            # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar = False)
        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)

        num_action = env.action_space.n
        num_state = env.observation_space.shape[0]
        reward_per_episode = []


        policy_dqn = DQN(num_state, num_action, self.fc1_nodes).to(device)
        
        if is_training:

            epsilon = self.epsilon_init

            memory = ReplayMemory(self.replay_memory_size)

            target_dqn = DQN(num_state, num_action, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            # Initialize the optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            epsilon_history = []
            
            # Initialize the target network and copy the weights from the policy network
            step_count = 0

            best_rewards = -9999999
        
        else:

            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        # Main training loop
        for episode in itertools.count():

            # Reset the environment
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device) #.unsqueeze(0)

            terminated = False
            episode_reward = 0.0

            while (not terminated and episode_reward < self.stop_on_reward):

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()
                
                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                
                # Accumulate reward
                episode_reward += reward
                
                #Convert new_state and reward to tensor
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device) #.unsqueeze(0)
                reward = torch.tensor(reward, dtype=torch.float32, device=device) #.unsqueeze(0)


                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1    

                state = new_state

            reward_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_rewards:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_rewards)/best_rewards*100:+.1f}%) at episode {episode}, saving model...\n"
                    print(log_message)
                    with open(self.LOG_FILE, "a") as file:
                        file.write(log_message)

                    best_rewards = episode_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(reward_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                if len(memory)>self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)     # decay epsilon
                    epsilon_history.append(epsilon)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

    def save_graph(self, reward_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(reward_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(reward_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def optimize(self, mini_batch, policy_dqn, target_dqn):

        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
        
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q,target_q)

        self.optimizer.zero_grad()      # reset the gradients
        loss.backward()                 # backpropagate to compute the gradients
        self.optimizer.step()           # update the model parameters based on the gradients and the optimizer




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters',help='')
    parser.add_argument("--train", help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameters_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)