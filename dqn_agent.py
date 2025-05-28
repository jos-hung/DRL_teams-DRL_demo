import torch 
from collections import deque
import numpy as np
import random
from config import device
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.000001, gamma=0.95, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, max_memory_size=10000, start_training=1024, batch_size=32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.epsilon_start
        self.max_memory_size = max_memory_size
        self.start_training = start_training  
        self.memory = deque()
        self.batch_size = batch_size
        # Initialize the Q-network
        self.q_network = self._build_model().to(device)
        self.target_q_network = self._build_model().to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()

    def _build_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.Linear(128, self.action_size),
            torch.nn.Tanh()
            )
        return model
    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory_size:
            self.memory.popleft()
            
    def get_actions(self, state):
        if torch.rand(1)[0] <  self.epsilon :  # Epsilon-greedy action selection
            action =  torch.randint(0, self.action_size, (1,)).item()
            # print(f"Random action: {action}, Epsilon: {self.epsilon:.2f}")
            return action
        with torch.no_grad():
            action = torch.argmax(self.q_network(state.to(device)).detach().cpu()[0]).item()
            # print(f"predict action: {action}, Epsilon: {self.epsilon:.2f}")
            return action
        
    
    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        print("Target network updated.")
    
    def update_model(self):
        if len(self.memory) < self.start_training:
            return

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        # Sample a mini-batch from memory
        mini_batch = random.sample(self.memory, self.batch_size) 

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in mini_batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_q_network(next_states) 
            next_q_value = next_q_values.max(dim=1)[0]

        target = rewards + (1 - dones) * self.gamma * next_q_value

        # MSE loss
        loss = self.criterion(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(f"Loss: {loss.item()}, Epsilon: {self.epsilon:.2f}")
