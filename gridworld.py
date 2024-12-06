import numpy as np

class Gridworld:
    def __init__(self, size=5):
        self.size = size
        self.grid = np.zeros((size, size))
        self.start_state = (0, 0)
        self.goal_state = (size-1, size-1)
        self.state = self.start_state
        self.obstacles = [(1, 1), (2, 2), (3, 3)] # Add obstacles here
        for obstacle in self.obstacles:
            self.grid[obstacle] = -1

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0 and x > 0:  # Up
            x -= 1
        elif action == 1 and x < self.size - 1:  # Down
            x += 1
        elif action == 2 and y > 0:  # Left
            y -= 1
        elif action == 3 and y < self.size - 1:  # Right
            y += 1

        self.state = (x, y)
        reward = 1 if self.state == self.goal_state else -0.01
        done = self.state == self.goal_state
        return self.state, reward, done
import random

class QLearningAgent:
    def __init__(self, env, lr=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.env = env
        self.q_table = np.zeros((env.size, env.size, 4))  # Four actions: up, down, left, right
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2, 3])
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])

    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        predict = self.q_table[x, y, action]
        target = reward + self.gamma * np.max(self.q_table[next_x, next_y])
        self.q_table[x, y, action] += self.lr * (target - predict)

    def train(self, episodes):
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
            self.epsilon *= self.epsilon_decay    
env = Gridworld()
agent = QLearningAgent(env)
agent.train(1000)

print("Training completed!")            

import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x
class PolicyGradientAgent:
    def __init__(self, env, lr=0.01):
        self.env = env
        self.policy_net = PolicyNetwork(2, 4) 
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = 0.99
    
    def choose_action(self, state):
        state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        action = np.random.choice(4, p=probs.detach().numpy()[0])
        return action

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            rewards = []
            actions = []
            states = []

            done = False
            while not done:
                states.append(state)
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                rewards.append(reward)
                actions.append(action)
                state = next_state
            
            # Compute discounted rewards
            discounted_rewards = []
            cumulative_reward = 0
            for reward in reversed(rewards):
                cumulative_reward = reward + self.gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)
            discounted_rewards = torch.FloatTensor(discounted_rewards)
            
            # Normalize rewards
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

            # Update policy network
            self.optimizer.zero_grad()
            for state, action, reward in zip(states, actions, discounted_rewards):
                state = np.array(state)
                state = torch.FloatTensor(state).unsqueeze(0)
                probs = self.policy_net(state)
                loss = -torch.log(probs[0, action]) * reward
                loss.backward()
            self.optimizer.step()
            print(f'Episode {episode + 1}/{episodes}, Loss: {loss.item()}')
env = Gridworld()
pg_agent = PolicyGradientAgent(env)
pg_agent.train(1000)

print("Policy Gradient Training completed!")

class QLearningAgent:
    def __init__(self, env, lr=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.env = env
        self.q_table = np.zeros((env.size, env.size, 4))  # Four actions: up, down, left, right
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2, 3])
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])

    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        predict = self.q_table[x, y, action]
        target = reward + self.gamma * np.max(self.q_table[next_x, next_y])
        self.q_table[x, y, action] += self.lr * (target - predict)

    def train(self, episodes):
        episode_rewards = []
        success_count = 0

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state)
                total_reward += reward
                state = next_state

            self.epsilon *= self.epsilon_decay
            episode_rewards.append(total_reward)
            if done and reward > 0:
                success_count += 1
        
        avg_reward = np.mean(episode_rewards)
        success_rate = success_count / episodes
        print(f"Q-Learning - Episodes: {episodes}, Average Reward: {avg_reward}, Success Rate: {success_rate * 100}%")
class PolicyGradientAgent:
    def __init__(self, env, lr=0.01):
        self.env = env
        self.policy_net = PolicyNetwork(2, 4)  # Input size should be 2 for (x, y) state
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = 0.99
    
    def choose_action(self, state):
        state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        action = np.random.choice(4, p=probs.detach().numpy()[0])
        return action

    def train(self, episodes):
        episode_rewards = []
        success_count = 0

        for episode in range(episodes):
            state = self.env.reset()
            rewards = []
            actions = []
            states = []

            done = False
            while not done:
                states.append(state)
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                rewards.append(reward)
                actions.append(action)
                state = next_state
            
            # Compute discounted rewards
            discounted_rewards = []
            cumulative_reward = 0
            for reward in reversed(rewards):
                cumulative_reward = reward + self.gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)
            discounted_rewards = torch.FloatTensor(discounted_rewards)
            
            # Normalize rewards
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

            # Update policy network
            self.optimizer.zero_grad()
            for state, action, reward in zip(states, actions, discounted_rewards):
                state = np.array(state)
                state = torch.FloatTensor(state).unsqueeze(0)
                probs = self.policy_net(state)
                loss = -torch.log(probs[0, action]) * reward
                loss.backward()
            self.optimizer.step()

            total_reward = sum(rewards)
            episode_rewards.append(total_reward)
            if done and reward > 0:
                success_count += 1
        
        avg_reward = np.mean(episode_rewards)
        success_rate = success_count / episodes
        print(f"Policy Gradient - Episodes: {episodes}, Average Reward: {avg_reward}, Success Rate: {success_rate * 100}%")
# Q-Learning Evaluation
env = Gridworld()
ql_agent = QLearningAgent(env)
ql_agent.train(1000)

# Policy Gradient Evaluation
pg_agent = PolicyGradientAgent(env)
pg_agent.train(1000)

import matplotlib.pyplot as plt

def plot_q_table(q_table):
    fig, ax = plt.subplots(1, figsize=(5, 5))
    for x in range(q_table.shape[0]):
        for y in range(q_table.shape[1]):
            if (x, y) in env.obstacles:
                ax.text(y, x, 'X', ha='center', va='center', color='red', fontsize=20)
            elif (x, y) == env.goal_state:
                ax.text(y, x, 'G', ha='center', va='center', color='green', fontsize=20)
            else:
                actions = ['↑', '↓', '←', '→']
                action = np.argmax(q_table[x, y])
                ax.text(y, x, actions[action], ha='center', va='center', color='black', fontsize=20)
    ax.set_xticks(np.arange(-0.5, q_table.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, q_table.shape[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='gray')
    plt.title("Learned Policy (Q-Learning)")
    plt.show()

# Visualize the Q-table
plot_q_table(ql_agent.q_table)
def plot_policy(policy_net, env):
    fig, ax = plt.subplots(1, figsize=(5, 5))
    for x in range(env.size):
        for y in range(env.size):
            if (x, y) in env.obstacles:
                ax.text(y, x, 'X', ha='center', va='center', color='red', fontsize=20)
            elif (x, y) == env.goal_state:
                ax.text(y, x, 'G', ha='center', va='center', color='green', fontsize=20)
            else:
                state = torch.FloatTensor([x, y]).unsqueeze(0)
                probs = policy_net(state).detach().numpy()[0]
                actions = ['↑', '↓', '←', '→']
                action = np.argmax(probs)
                ax.text(y, x, actions[action], ha='center', va='center', color='black', fontsize=20)
    ax.set_xticks(np.arange(-0.5, env.size, 1))
    ax.set_yticks(np.arange(-0.5, env.size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='gray')
    plt.title("Learned Policy (Policy Gradient)")
    plt.show()

# Visualize the policy
plot_policy(pg_agent.policy_net, env)
