# ============================================================================
# IMPORTS
# ============================================================================

from autonomous_decision_system import Autonomous_Decision_System
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import random as rnd
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# ============================================================================

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
print('Using device:', device)

# ============================================================================

# DQN:
class Environment():
    def __init__(self, states, actions, subscriber=None):
        self.states = states
        self.actions = actions
        self.state_dim = len(self.states[0])
        self.num_actions = len(self.actions)
        self.subscriber = subscriber
        self.reset()

    def reset(self):
        i = random.choice(range(len(self.states)))
        self.state = self.states[i]  # random initial state

    def execute_action(self, action):
        self.state = self.actions[action]
        res = self.subscriber.update(self.state) if self.subscriber else 1  # Avoid undefined behavior
        
        # Example of reward calculation based on multiple metrics
        distance = self.state[0]  # Assume distance is stored in the first element of the state
        time_taken = self.state[1]  # Assume time is stored in the second element of the state
        
        # Calculate reward based on distance and time
        self.reward = 1 / (1 + distance + time_taken)  # Reward between 0 and 1
        return self.reward


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim

        self.ff = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(32, self.output_dim),
        )
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        qvals = self.ff(state)
        return qvals


class Buffer():
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def size(self):
        return len(self.buffer)

    def push(self, state, action, new_state, reward):
        experience = (state, action, new_state, reward)
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)  # Remove oldest experience
        self.buffer.append(experience)

    def sample(self, batch_size):
        batchSample = random.sample(self.buffer, batch_size)
        state_batch, action_batch, new_state_batch, reward_batch = zip(*batchSample)
        return (list(state_batch), list(action_batch), list(reward_batch), list(new_state_batch))


class DeepAgent():
    def __init__(self, state_dim, num_actions, episodes):
        self.policy_net = DQN(state_dim, num_actions).to(device)
        self.target_net = DQN(state_dim, num_actions).to(device)
        self.target_net.eval()
        self.target_update = 1000
        self.replay_buffer = Buffer()
        self.eps_start = 1
        self.eps_end = 0.1
        self.limit_episodes = episodes
        self.eps_decay = 0.99
        self.epsilon = self.eps_start
        self.gamma = 0.9  # Adjusted gamma to a more typical value
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate)
        self.huber_loss = nn.MSELoss()

    def select_action(self, state, num_actions):
        state = torch.FloatTensor(state).float().to(device)
        if rnd.rand() < (1 - self.epsilon):
            with torch.no_grad():
                qvals = self.policy_net.forward(state)
                action = np.argmax(qvals.cpu().detach().numpy())
        else:
            action = random.choice(list(range(num_actions)))
        return action

    def update(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return None
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        return loss

    def compute_loss(self, batch):
        states, actions, rewards, next_states = batch
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        curr_Q = self.policy_net.forward(states).gather(1, actions.unsqueeze(1))
        next_Q = self.target_net.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards + self.gamma * max_next_Q
        loss = self.huber_loss(curr_Q, expected_Q.unsqueeze(1))
        return loss


class DeepRLInterface():
    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment
        self.batch_size = 50
        self.writer = SummaryWriter()  # Initialize the writer attribute
        self.rewlist = []  # Initialize rewlist
        self.losslist = []  # Initialize losslist



    def step(self, num_actions):
        state = self.env.state.copy()
        action = self.agent.select_action(state, num_actions)
        rew = self.env.execute_action(action)
        new_state = self.env.state.copy()
        self.agent.replay_buffer.push(state, action, new_state, rew)
        loss = self.agent.update(self.batch_size)
        if loss is not None:
            self.losslist.append(loss.item())
        return state, action, rew, new_state

    def runTrials(self, nTrials, steps, num_actions):
        counter = 0
        self.rewlist = []
        self.losslist = []
        for i in range(nTrials):
            self.env.reset()
            total_rew = 0
            for j in range(steps):
                state, action, rew, new_state = self.step(num_actions)
                total_rew += rew
                print("Episode:", i, "Step:", j)
                print("State:", new_state, "Action:", action)
                print("Reward:", rew, "Epsilon:", self.agent.epsilon)
                counter += 1

            self.rewlist.append(total_rew)

            if counter % self.agent.target_update == 0:
                self.agent.target_net.load_state_dict(
                    self.agent.policy_net.state_dict())

            self.agent.epsilon = max(self.agent.eps_end, self.agent.eps_start - (
                i * self.agent.eps_decay))

            self.writer.add_scalar('Accumulated reward per episode',
                                   total_rew, i)
            self.writer.add_scalar('Epsilon', self.agent.epsilon, i)

            if (i + 1) % 100 == 0:
                PATH = (f"model-12-{i}.pt")
                torch.save({
                    'epoch': i,
                    'model_state_dict': self.agent.target_net.state_dict(),
                    'optimizer_state_dict':
                    self.agent.optimizer.state_dict(),
                }, PATH)


class RL_AI_MODEL(Autonomous_Decision_System):

    def __init__(self):
        Autonomous_Decision_System.__init__(self)

        # number of episodes and steps
        self.max_episodes = 500
        self.max_steps = 100

        # initialize states and actions
        e1 = np.arange(60, 310, 10)
        e2 = np.repeat(e1, 25)
        e3 = np.arange(10, 60, 10)
        e4 = np.tile(e3, 125)
        e5 = np.arange(1, 6, 1)
        e6 = np.repeat(e5, 5)
        e7 = np.tile(e6, 25)
        e8 = np.column_stack((e2, e4))
        self.S = np.column_stack((e8, e7))  # 625 states
        self.actions = np.column_stack((e8, e7))  # 625 actions

        # initialize agent and environment
        self.env = Environment(states=self.S, actions=self.actions)
        self.n_states = self.env.state_dim
        self.n_actions = self.env.num_actions
        self.agent = DeepAgent(self.n_states, self.n_actions,
                                self.max_episodes)
        # Initialize rewlist and losslist
        self.rewlist = []
        self.losslist = []

    # RL function - update states and Q matrix
    def process(self):
        self.writer = SummaryWriter()
        try:
            rl = DeepRLInterface(self.agent, self.env)
            rl.runTrials(self.max_episodes, self.max_steps, self.n_actions)
            self.rewlist = rl.rewlist  # Copy lists from the interface
            self.losslist = rl.losslist
        finally:
            self.writer.close()

    def plot(self):
        # Plot rewards and losses
        plt.figure(figsize=(12, 5))
        
        # Plot rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.rewlist)
        plt.title('Accumulated reward per episode')
        plt.xlabel('Episode')
        plt.ylabel('Accumulated reward')

        # Plot losses
        plt.subplot(1, 2, 2)
        plt.plot(self.losslist)
        plt.title('Loss per episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

        plt.tight_layout()
        # Save the image
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, "plot_results.png")
        plt.savefig(save_path)  # Save plots in the same directory
        print(f"Results have been saved in {save_path}")
        plt.show()