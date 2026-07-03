import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random
import ale_py
import gymnasium as gym
from gymnasium import spaces
import cv2
import math
import random
from collections import deque
import matplotlib.pyplot as plt
# from google.colab import drive
# drive.mount('/content/gdrive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ImageProcessing:
    def __init__(self,screen):
        self.screen = screen
    def rgb_to_grayscale(self):
        # m,n,x = self.screen.shape
        # new_screen = np.zeros((m,n),dtype = np.uint8)
        # for i in range(m):
        #     for j in range(n):
        #         r,g,b = self.screen[i,j]
        #         new_screen[i,j] = 0.299*r+ 0.587*g + 0.114*b
        # self.screen = new_screen
        self.screen = cv2.cvtColor(self.screen, cv2.COLOR_RGB2GRAY)

    def resize_frame(self,target_size = tuple([84, 84])):
        self.screen = cv2.resize(self.screen,target_size,interpolation= cv2.INTER_AREA)

    def normalize_frame(self):
        self.screen = self.screen.astype(np.float32)
        self.screen = self.screen/255

    def preprocess_frame(self):
        self.rgb_to_grayscale()
        self.resize_frame()
        self.normalize_frame()
class FrameStack:
    def __init__(self, maxlen):
        """Initialize deque with maxlen capacity"""
        # Set up a data structure to store frames with automatic size management.
        # Store the maximum length for reference in other methods.
        self.stack = deque([],maxlen = maxlen)
        self.maxlen = maxlen
        self.curlen = 0
        pass

    def push(self,frame):
        """Add preprocessed frame to deque"""
        # Add the new frame to the collection.
        # The data structure should automatically handle overflow.
        self.stack.append(frame)
        self.curlen += 1
        pass

    def get_stack(self):
        """Return stacked frames as (maxlen, height, width)"""
        # Create a properly shaped array containing all frames.
        # Handle cases where fewer frames are available by padding with zeros.
        # Return frames in channel-first format for CNN input.
        return np.stack(list(self.stack))

    def reset(self):
        """Clear the deque"""
        # Remove all stored frames to start fresh.
        self.stack.clear()
        self.curlen = 0
class AtariWrapper:
    def __init__(self,env_name = "BreakoutNoFrameskip-v4",frame_skip = 4):
        """Initialize environment and frame stack"""
        # Create the Gym environment and frame stack manager.
        # Define the action space mapping for Breakout game.
        # Store frame skipping parameter for temporal efficiency.
        self.env = gym.make(env_name)
        self.stack = FrameStack(maxlen=4)
        self.frame_skip = frame_skip
        observation,info = self.env.reset()

    def reset(self):
        # Reset both the environment and frame stack.
        # Process the initial observation and create the first state stack.
        # Return the properly formatted state for the agent.
        observation,info = self.env.reset()
        self.stack.reset()
        obj = ImageProcessing(observation)
        obj.preprocess_frame()
        frame = obj.screen
        for _ in range(self.stack.maxlen):
            self.stack.push(frame)
        return self.get_state()

    def step(self, action):
        """Execute action with frame skipping"""
        # Convert agent action to environment action.
        # Execute the action multiple times to skip frames.
        # Accumulate rewards and process the final frame.
        # Return the new state, total reward, done flag, and info.
        total_reward = 0
        for _ in range(self.frame_skip):  # Skip 4 frames but store only the last one
            observation, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated: break

        # Process only the final frame
        obj = ImageProcessing(observation)
        obj.preprocess_frame()
        self.stack.push(obj.screen)  # Add 1 new frame to the stack

        return self.get_state(), total_reward,truncated, done, info

    def get_state(self):
        """Return current stacked state"""
        # Return the current frame stack as a state representation.
        return self.stack.get_stack()

    def close(self):
        """Close the environment"""
        # Properly close the Gym environment to free resources.
        self.env.close()
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_ptr = 0
        self.size = 0
    def _propagate(self, tree_idx: int, change: float) -> None:
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    def add(self, priority, data):
        tree_idx = self.data_ptr + self.capacity - 1
        self.data[self.data_ptr] = data
        self.update(tree_idx, priority)

        self.data_ptr += 1
        if self.data_ptr >= self.capacity:
            self.data_ptr = 0
        self.size += 1
        self.size = min(self.size,self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx,change)
    def get_leaf(self, s):
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1

            if left >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if s <= self.tree[left]:
                    parent_idx = left
                else:
                    s -= self.tree[left]
                    parent_idx = right

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha, beta_start):
        """Initialize prioritized replay buffer"""
        # Set up the SumTree and store prioritization parameters.
        # Initialize beta annealing schedule and numerical stability constants.
        # Set up frame counting for importance sampling weight calculation.
        self.SumTree = SumTree(capacity)
        self.frame = 0
        self.alpha = alpha
        self.beta = beta_start
        self.capacity = capacity
        self.epsilon = 0.001
        self.max_priority = 1

    def _get_beta(self) -> float:
        """Calculate current beta value (anneals to 1.0)"""
        # Implement linear annealing from beta_start to 1.0 over training.
        self.frame += 1
        self.beta = min(1.0, self.beta + self.frame * 0.001)
        return self.beta

    def push(self, state, action, reward, next_state, done) -> None:
        """Store experience with maximum priority"""
        # Package the experience tuple and assign appropriate priority.
        # Use maximum existing priority for new experiences to ensure sampling.
        self.SumTree.add(self.max_priority,(state,action,reward,next_state,done))

    def sample(self, batch_size: int) -> tuple[list, np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights"""
        # Divide priority range into segments for stratified sampling.
        # Calculate importance sampling weights to correct for sampling bias.
        # Return experiences, tree indices, and normalized weights.
        batch = []
        idxs = []
        segment = self.SumTree.total_priority() / batch_size
        priorities = []

        self.beta = self._get_beta()

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.SumTree.get_leaf(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.SumTree.total_priority()
        is_weights = np.power((self.SumTree.size) * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update_priorities(self, indices, priorities) -> None:
        """Update priorities based on TD errors"""
        # Convert TD errors to priorities using alpha exponent.
        # Add small epsilon for numerical stability.
        # Update tree nodes with new priority values.
        for idx, priority in zip(indices, priorities):
            priority = (abs(priority) + self.epsilon) ** self.alpha
            self.SumTree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


    def __len__(self) -> int:
        """Return current buffer size"""
        # Return the number of experiences currently stored.
        return self.SumTree.size
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        # Register buffers for noise (non-learnable)
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.mu_bias.data.uniform_(-mu_range, mu_range)
        self.sigma_weight.data.fill_(0.017)  # recommended: sigma_init / sqrt(in_features)
        self.sigma_bias.data.fill_(0.017)

    def reset_noise(self):
        epsilon_in = self._f(torch.randn(self.in_features)).to(self.mu_weight.device)
        epsilon_out = self._f(torch.randn(self.out_features)).to(self.mu_weight.device)
        self.epsilon_input = epsilon_in.unsqueeze(0)
        self.epsilon_output = epsilon_out.unsqueeze(1)

    def forward(self, x):
        if self.training:
            weight = self.mu_weight + self.sigma_weight * (self.epsilon_output @ self.epsilon_input)
            bias = self.mu_bias + self.sigma_bias * self.epsilon_output.squeeze()
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return torch.nn.functional.linear(x, weight, bias)

    def _f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))
class DDQN(nn.Module):
    def __init__(self, input_dim: tuple[int, int, int], action_dim: int):
        """Initialize DDQN with convolutional layers"""
        # Set up the parent class and define three convolutional layers.
        # Use progressively smaller kernels and increasing channels.
        # Calculate the flattened size after convolutions for fully connected layers.
        # Add two fully connected layers to map features to Q-values.
        super(DDQN,self).__init__()  
        # self.layer1 = nn.Sequential(nn.Conv2d(in_channels=4,out_channels=16,kernel_size=8,stride = 4),nn.ReLU())
        # self.layer2 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride = 2),nn.ReLU())
        # self.layer3 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride = 1),nn.ReLU())
        self.CNN = nn.Sequential(nn.Conv2d(in_channels=4,out_channels=16,kernel_size=8,stride = 4),nn.ReLU(),nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride = 2),nn.ReLU(),nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride = 1),nn.ReLU())
        conv_out = self._get_conv_out_size(input_dim)
        flat = np.prod(conv_out)
        # self.layer4 = nn.Sequential(nn.Linear(flat,256),nn.ReLU())
        # self.layer5 = nn.Sequential(nn.Linear(256,action_dim))
        self.values = nn.Sequential(NoisyLinear(flat, 512), nn.ReLU(),NoisyLinear(512, 1))
        self.advantages = nn.Sequential(NoisyLinear(flat, 512), nn.ReLU(),NoisyLinear(512, action_dim))
    def _get_conv_out_size(self, shape: tuple[int, int, int]) -> int:
        """Calculate output size of convolutional layers"""
        # Pass a dummy tensor through the convolutional layers.
        # Calculate the total number of features after flattening.
        test = torch.randn(1,*shape)
        test = self.CNN(test)
        return test.shape
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.CNN(x)
        x = x.view(x.size(0), -1)
        values = self.values(x)
        advantages = self.advantages(x)
        Q = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return Q
    def reset_noise(self):
        for m in self.advantages:
            if isinstance(m, NoisyLinear):
                m.reset_noise()
def compute_double_DDQN_loss(policy_net, target_net, states, actions,
                              rewards, next_states, dones, gamma, is_weights):
    """Compute Double DDQN loss with importance sampling"""

    # Extract Q-values for the actions that were actually taken.
    # Use the policy network to select the best next actions.
    # Evaluate those selected actions using the target network.
    # Compute target Q-values using the Bellman equation.
    # Calculate temporal difference errors for priority updates.
    # Apply importance sampling weights if provided to correct sampling bias.
    # Return both the loss value and TD errors for priority updates.
    # states = torch.tensor(states, dtype=torch.float32).to(device)
    # actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    # rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    # next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    # dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    # is_weights = torch.tensor(is_weights, dtype=torch.float32).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)

    next_q_values = target_net(next_states).gather(1, next_actions)

    target_q = rewards + gamma * next_q_values * (1 - dones)

    td_errors = target_q - q_values

    loss = (td_errors.pow(2) * is_weights).mean()

    return loss, td_errors.detach().squeeze()
class AdvancedDDQNAgent:
    def __init__(self, state_shape, action_size, config):
        """Initialize agent with networks and replay buffer"""
        # Store network dimensions and configuration parameters.
        # Create policy and target networks with identical architectures.
        # Initialize the target network with policy network weights.
        # Set up the optimizer and prioritized replay buffer.
        # Configure exploration parameters for epsilon-greedy strategy.

        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.capacity = config['buffer_size']
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        self.initial_replay_size = config['initial_replay_size']
        self.beta_start = config['beta_start']
        self.max_episodes = config['max_episodes']
        self.state_shape = state_shape
        self.action_size = action_size
        self.policy_network = DDQN(state_shape,action_size).to(device)
        self.target_network = DDQN(state_shape,action_size).to(device)
        self.optimizer = optim.AdamW(self.policy_network.parameters(),lr = self.learning_rate)
        self.buffer = PrioritizedReplayBuffer(self.capacity,self.alpha,self.beta_start)
        self.epsilon = 0
        self.atari = AtariWrapper()
        self.losses = []
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        # Calculate current epsilon value using exponential decay schedule.
        # Choose between random exploration and greedy exploitation.
        # For greedy actions, use the policy network to select best action.
        self.epsilon = max(0,self.epsilon*0.999)
        if np.random.random()<self.epsilon:
            return self.atari.env.action_space.sample()
        state_tensor = state.detach().clone().unsqueeze(0).to(device)
        with torch.no_grad():
            return self.policy_network(state_tensor).argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        # Add the experience tuple to the prioritized replay buffer.
        self.buffer.push(state,action,reward,next_state,done)

    def update(self):
        """Perform learning update"""
        # Check if sufficient experiences are available for training.
        # Sample a batch from the prioritized replay buffer.
        # Convert experience components to tensors and move to device.
        # Compute Double DDQN loss and TD errors.
        # Update experience priorities based on TD errors.
        # Perform gradient descent optimization step.
        if self.buffer.__len__() > self.initial_replay_size:
            batch,index,weights = self.buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.from_numpy(np.array(states)).float().to(device)
            actions     = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(device)
            rewards     = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(device)
            next_states = torch.from_numpy(np.array(next_states)).float().to(device)
            dones       = torch.from_numpy(np.array(dones)).float().unsqueeze(1).to(device)
            weights     = torch.from_numpy(np.array(weights)).float().unsqueeze(1).to(device)
            loss,TD = compute_double_DDQN_loss(self.policy_network,self.target_network,states,actions,rewards,next_states,dones,self.gamma,weights)
            priorities = TD.abs().cpu().numpy()
            self.buffer.update_priorities(index,priorities)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())
    def update_target_network(self):
        """Copy weights from policy to target network"""
        # Synchronize target network weights with policy network.
        with torch.no_grad():
            self.target_network.load_state_dict(self.policy_network.state_dict())
config = {
    'learning_rate': 3e-4,  # Learning rate for the optimizer
    'gamma': 0.99,  # Discount factor for future rewards
    'buffer_size': 50000,  # Maximum size of the replay buffer
    'batch_size': 64,  # Number of samples to draw from the buffer for each training step
    'target_update_freq': 500,  # Frequency (in training steps) to update the target network
    'initial_replay_size':10000,  # Minimum number of experiences in the buffer before training starts
    'alpha': 0.6,  # PER prioritization
    'beta_start': 0.4,  # PER importance sampling
    'max_episodes': 2000,  # Maximum number of episodes to run
    'target_score': 12.0,  # Mean score over 50 episodes
    'V_MAX' : 10,
    'V_MIN' : -10,
    'N_ATOMS' : 51
}
V_MIN = config['V_MIN']
V_MAX = config['V_MAX']
DELTA_Z = (V_MAX - V_MIN) / (config['N_ATOMS'] - 1)
SUPPORT = torch.linspace(V_MIN, V_MAX, config['N_ATOMS'])
def projection_distribution(next_dist, rewards, dones, gamma):
    batch_size = rewards.size(0)
    proj_dist = torch.zeros((batch_size, config['N_ATOMS'])).to(device)

    for i in range(config['N_ATOMS']):
        tz_j = torch.clamp(rewards + (1 - dones) * gamma * SUPPORT[i], V_MIN, V_MAX)
        b_j = (tz_j - V_MIN) / DELTA_Z
        l = b_j.floor().long()
        u = b_j.ceil().long()

        eq_mask = (u == l)

        proj_dist[range(batch_size), l] += next_dist[:, i] * (u - b_j)
        proj_dist[range(batch_size), u] += next_dist[:, i] * (b_j - l)
        proj_dist[eq_mask, l[eq_mask]] += next_dist[eq_mask, i]

    return proj_dist
steps = 0
def train_agent():
    """Main training loop with comprehensive monitoring"""
    # Initialize the Atari environment wrapper and DDQN agent.
    # Set up tracking variables for metrics and timing.
    # Implement the main episode loop with proper environment interaction.
    # Handle experience storage, agent updates, and target network synchronization.
    # Monitor training progress with comprehensive logging and statistics.
    # Implement early stopping when target performance is achieved.
    # Return training metrics and the trained agent for analysis.
    agent = AdvancedDDQNAgent((4, 84, 84), 4, config)
    agent.policy_network.train()
    agent.target_network.train()
    Episode_Rewards = []
    mean_scores = []
    losses = []
    window_size = 50
    global steps
    for episode in range(config['max_episodes']):
        obs = agent.atari.reset()
        
        done = False
        rewards = 0
        curr = obs
        obs,rewards,truncated,done,info = agent.atari.step(1)
        while not done:
            agent.policy_network.reset_noise()
            agent.target_network.reset_noise()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            action = agent.select_action(obs_tensor)

            next_obs, reward,truncated, done, info = agent.atari.step(action)
            rewards += reward
            if truncated: reward -= 2.5
            # if done : reward += 0.1*(reward-50)
            agent.store_transition(obs, action, reward, next_obs, done)

            if len(agent.buffer) > config['initial_replay_size']:
                agent.update()
                steps += 1
                if steps % config['target_update_freq'] == 0:
                    agent.update_target_network()
            curr = obs
            obs = next_obs
        if episode+1 >= window_size:
            mean_score = np.mean(Episode_Rewards[-window_size:]) if Episode_Rewards[-window_size:] else 0
        else:
            mean_score = np.mean(Episode_Rewards) if Episode_Rewards else 0
        mean_scores.append(mean_score)
        Episode_Rewards.append(rewards)
        print(f'Episode No. : {episode+1}, reward = {rewards}')
        losses = agent.losses
    torch.save(agent.target_network, 'my_model_v2.pth')
    plot_training_results(Episode_Rewards,mean_scores,losses)
def plot_training_results(episode_rewards, mean_scores, losses):
    """Create comprehensive training visualization"""
    # Set up a multi-panel figure to display various training metrics.
    # Plot episode rewards with rolling mean to show learning progress.
    # Display mean score progression with target achievement line.
    # Show training loss evolution to monitor learning stability.
    # Create score distribution histogram to analyze performance spread.
    # Add proper labeling, legends, and save the results.
    if len(mean_scores) < len(episode_rewards):
        pad_length = len(episode_rewards) - len(mean_scores)
        mean_scores += [mean_scores[-1]] * pad_length
    if len(losses) < len(episode_rewards):
        losses += [0.0] * (len(episode_rewards) - len(losses))

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    episodes = np.arange(1, len(episode_rewards)+1)

    axs[0, 0].plot(episodes, episode_rewards)
    axs[0, 0].plot(episodes, mean_scores, label='Mean Score (50 eps)')
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].legend()

    axs[0, 1].plot(episodes, mean_scores)
    axs[0, 1].axhline(y=config['target_score'], linestyle='dotted', c='r', label='Target Score')
    axs[0, 1].set_title('Mean Score Progression')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Mean Score (50 Episodes)')
    axs[0, 1].legend()


    axs[1, 0].plot(np.arange(1,len(losses)+1), losses)
    axs[1, 0].set_title('Training Loss')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Loss')

    axs[1, 1].hist(episode_rewards,width = 0.3, alpha=0.7)
    axs[1,1].set_xticks(np.arange(min(episode_rewards), max(episode_rewards)+1, 1))
    axs[1, 1].axvline(x=np.mean(episode_rewards), color='r', linestyle='dashed',
    label=f'Mean: {np.mean(episode_rewards):.1f}')
    axs[1, 1].set_title('Reward Distribution')
    axs[1, 1].set_xlabel('Reward')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
if __name__ == "__main__":
    """Main execution block for the assignment"""
    # Display assignment title and information.
    # Execute the training process and collect results.
    # Generate and display comprehensive visualizations.
    # Print completion message and file references.
    train_agent()
    print('Completed')

