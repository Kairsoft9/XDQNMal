import torch
import torch.nn.functional as F
import random
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from collections import deque
import torch.nn.init as init
import torch.nn.utils as nn_utils


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        init.constant_(m.bias, 0)


class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        # self.attention = Attention(state_dim)
        # self._initialize_weights()
        self.apply(init_weights)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        out = self.fc3(x)
        return out




class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, zero_mask):
        self.buffer.append((state, action, reward, next_state, done, zero_mask))

    def sample(self, batch_size):
        state, action, reward, next_state, done, zero_mask = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done, zero_mask

    def sample_all(self):
        state, action, reward, next_state, done, zero_mask = zip(*list(self.buffer))
        return state, action, reward, next_state, done, zero_mask

    def __len__(self):
        return len(self.buffer)
        return len(self.buffer)


class DQN:
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, epsilon, gamma, lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.model = DQNNet(state_dim, action_dim).to(self.device)
        self.target_model = DQNNet(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.count = 0
        self.dynamic_mask = np.ones(self.action_dim)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            with (torch.no_grad()):
                state = torch.tensor(state, dtype=torch.float).to(self.device)
                output = self.model(state.unsqueeze(0))
                action_values = output.cpu().detach().numpy().squeeze()
                if self.dynamic_mask is not None:
                    action_values[self.dynamic_mask == 0] = -np.inf
                action = np.argmax(action_values)
        # print(action)
        return action

    def update(self, transtition_dict):
        states = torch.tensor(np.array(transtition_dict['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(transtition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transtition_dict['rewards']), dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transtition_dict['next_states']), dtype=torch.float32).to(self.device)
        dones = torch.tensor(transtition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)
        zero_masks = torch.tensor(np.array(transtition_dict['zero_masks']), dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        q_values = self.model(states)
        q_values = q_values.gather(1, actions)
        next_q_values = self.target_model(next_states)
        next_q_values = next_q_values.max(1)[0].view(-1, 1)
        expected_q_value = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.smooth_l1_loss(q_values, expected_q_value.detach())
        total_loss = loss + penalty
        total_loss.backward()
        nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)


        self.optimizer.step()

        if self.count % 20 == 0:
            self.update_target()
        self.count += 1


    def get_key_features_withQ(self):
        fc3_weights = self.model.fc3.weight.data.cpu().numpy()
        feature_impacts = fc3_weights.sum(axis=0)
        key_features = np.argsort(feature_impacts)[::-1]
        return key_features, feature_impacts

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def reset_dynamic_mask(self):
        self.dynamic_mask = np.ones(self.action_dim)

    def save_model(self, file_name='dqn_model_f.pth'):
        torch.save(self.model.state_dict(), file_name)
        print(f"Model saved to {file_name}")

    def load_model(self, file_name='dqn_model_f.pth'):
        self.model.load_state_dict(torch.load(file_name))
        self.target_model.load_state_dict(torch.load(file_name))
        print(f"Model loaded from {file_name}")


class Environment:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.previous_accuracy = self.model.predict_proba(self.X[0].reshape(1, -1))[:, self.y[0]]
        self.state = self.X[0]
        self.mask = np.ones_like(self.state)
        self.count = 1
        self.num = 0
        self.list = []
        self.count_num = []
        self.accuracyx = []

    def step(self, actions, label):
        next_states = []
        final_states = []
        done = false
        rewards = []
  
        return final_states, rewards, done

    def step2(self, actions, label):
        next_states = []
        final_states = []
        rewards = []
        done = false

            
        return final_states, rewards, done

    def reset(self, index):
        self.state = self.X[index]
        self.count = 1
        self.num = 0
        self.mask = np.ones_like(self.state)
        self.previous_accuracy = self.model.predict_proba(self.state.reshape(1, -1))[:, self.y[index]]
        self.list = []
        self.count_num = []
        self.accuracyx = []
        return self.state
