from autonomous_decision_system import Autonomous_Decision_System
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import time


class DQN(nn.Module):
    def __init__(self, envstate_dim, action_dim):
        super(DQN, self).__init__()
        self.input_dim = envstate_dim
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

class RL_Method_3(Autonomous_Decision_System):

    def __init__(self, pretrained_model_path):
        super().__init__()

        self.gamma = 0.5
        self.initial_epsilon = 1
        self.minimum_epsilon = 0.1
        self.limit_episodes = 229
        self.epsilon_decay = 0.99

        # Number of episodes and steps
        self.max_episodes = 500
        self.max_steps = 100

        # Initialize rewards per episode
        self.episode_reward = np.zeros(self.max_episodes)

        # Define states and actions
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

        # Initialize epsilon
        self.epsilon = self.initial_epsilon

        # Load the pretrained model
        self.pretrained_model = self.load_pretrained_model(pretrained_model_path)

    def load_pretrained_model(self, model_path):
        """
        Load the pretrained model from the given path.
        """
        model = DQN(self.S.shape[1], self.actions.shape[0])
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def select_action(self, state):
        """
        Use the pretrained model to select an action based on the current state.
        """
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Get action probabilities or values from the model
        with torch.no_grad():
            action_values = self.pretrained_model(state_tensor)

        # Choose the action with the highest predicted value
        action_index = torch.argmax(action_values).item()
        return action_index

    def process(self):
        writer = SummaryWriter()
        for n in range(self.max_episodes):
            S0 = self.S[0]  # Start state
            t = 0
            r_acum = 0
            res0 = self.subscriber.update(S0)
            r_tot = res0
            

            while t < self.max_steps:
                print("Episode", n, "Step", t)
                # Select action using pretrained model
                action_index = self.select_action(S0)
                Snew = self.S[action_index]  # Use the selected action to get the new state
                
                # Update simulation result
                res1 = self.subscriber.update(Snew)

               # Compute reward
                distance = Snew[0]  # Assume distance is stored in the first element of the state
                time_taken = Snew[1]  # Assume time is stored in the second element of the state
                r = 1 / (1 + distance + time_taken)  # Reward between 0 and 1

                # Update parameters
                t += 1
                S0 = Snew
                r_acum += r
                r_tot += res1
              

            self.episode_reward[n] = r_acum

            # Adjust epsilon for exploration (if applicable)
            if n >= self.limit_episodes:
                self.epsilon = self.minimum_epsilon
            else:
                self.epsilon *= self.epsilon_decay

            # Log metrics to TensorBoard
            writer.add_scalar('Accumulated reward per episode', r_acum, n)
            writer.add_scalar('Average reward', r_acum / self.max_steps, n)
            writer.add_scalar('Epsilon', self.epsilon, n)
            writer.add_scalar('Total result to minimize', r_tot, n)
            writer.add_scalar('Average result to minimize', r_tot / self.max_steps, n)