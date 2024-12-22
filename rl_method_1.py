# ============================================================================
# IMPORTS
# ============================================================================

from autonomous_decision_system import Autonomous_Decision_System
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# ============================================================================


# Q-Learning:
class RL_Method_1(Autonomous_Decision_System):

    def __init__(self):
        Autonomous_Decision_System.__init__(self)

        self.alpha = 0.9
        self.gamma = 0.5
        self.initial_epsilon = 1
        self.minimum_epsilon = 0.1
        self.limit_episodes = 229
        self.epsilon_decay = 0.99

        # number of episodes and steps
        self.max_episodes = 500
        self.max_steps = 100

        # initialize reward per episode
        self.episode_reward = np.arange(self.max_episodes, dtype=float)

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

        # initialize Q table
        self.Q = np.zeros((self.S.shape[0], self.actions.shape[0]))

        # Initialize epsilon
        self.epsilon = self.initial_epsilon

    # function to choose action
    def choose_action(self, row):
        p = np.random.random()
        if p < (1 - self.epsilon):
            i = np.argmax(self.Q[row, :])
        else:
            i = np.random.choice(2)
        return i

    # RL function - update states and Q matrix
    def process(self):
        writer = SummaryWriter()
        for n in range(self.max_episodes):
            S0 = self.S[0]
            t = 0
            r_acum = 0
            res0 = self.subscriber.update(S0)
            r_tot = res0
            while t < self.max_steps:
                print("Episode", n, "Step", t)
                # find index k of the current state
                for k in range(625):
                    if self.S[k][0] == S0[0] and self.S[k][1] == S0[1] and self.S[k][2] == S0[2]:
                        break
                # choose action from row k
                j = self.choose_action(k)
                # update state
                Snew = self.actions[j]
                # update simulation result
                res1 = self.subscriber.update(Snew)
                # reward
                r = 1 / res1
                # find index of the new state S'
                for l in range(625):
                    if self.S[l][0] == Snew[0] and self.S[l][1] == Snew[1] and self.S[l][2] == Snew[2]:
                        break
                # update Q matrix
                self.Q[k, j] = self.Q[k, j] + self.alpha * (r + self.gamma * np.max(self.Q[l, :]) - self.Q[k, j])
                # update parameters
                t += 1
                S0 = Snew
                r_acum = r_acum + r
                r_tot = r_tot + res1
            self.episode_reward[n] = r_acum
            if n >= self.limit_episodes:
                self.epsilon = self.minimum_epsilon
            else:
                self.epsilon = self.epsilon * self.epsilon_decay
            writer.add_scalar('Accumulated reward per episode', r_acum, n)
            writer.add_scalar('Average reward', r_acum / self.max_steps, n)
            writer.add_scalar('Epsilon', self.epsilon, n)
            writer.add_scalar('Total result to minimize', r_tot, n)
            writer.add_scalar('Average result to minimize', r_tot / self.max_steps, n)