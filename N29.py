import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for Noisy Nets.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters for weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Register buffers for noise
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Initialize the parameters.
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    def reset_noise(self):
        """
        Sample new noise.
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, output_dim, atom_size=51, v_min=-10, v_max=10):
        """
        RAINBOW DQN Model.
        Args:
            input_shape (tuple): Shape of the input (rows, cols) for the board.
            output_dim (int): Number of possible actions (columns for Connect Four).
            atom_size (int): Number of atoms for the categorical distribution.
            v_min (float): Minimum value in the support.
            v_max (float): Maximum value in the support.
        """
        super(RainbowDQN, self).__init__()

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max

        # Support for Distributional RL
        self.register_buffer("support", torch.linspace(self.v_min, self.v_max, self.atom_size))

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Calculate the size of the flattened output after convolutions
        conv_output_size = 128 * input_shape[0] * input_shape[1]

        # Fully connected layers for dueling architecture
        self.fc_value = nn.Linear(conv_output_size, 256)
        self.noisy_value = NoisyLinear(256, atom_size)
        self.fc_advantage = nn.Linear(conv_output_size, 256)
        self.noisy_advantage = NoisyLinear(256, output_dim * atom_size)

    def forward(self, x):
        """
        Forward pass of the RAINBOW DQN.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, rows, cols).
        Returns:
            torch.Tensor: Distributional Q-values for each action.
        """
        # Reshape input to (batch_size, 1, rows, cols) for CNN
        x = x.view(-1, 1, *self.input_shape)

        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Compute value and advantage streams
        value = F.relu(self.fc_value(x))
        value = self.noisy_value(value).view(-1, 1, self.atom_size)  # Shape: (batch_size, 1, atom_size)

        advantage = F.relu(self.fc_advantage(x))
        advantage = self.noisy_advantage(advantage).view(-1, self.output_dim, self.atom_size)  # Shape: (batch_size, output_dim, atom_size)

        # Combine streams using dueling architecture
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Apply softmax to output a probability distribution
        q_atoms = F.softmax(q_atoms, dim=-1)

        return q_atoms

    def reset_noise(self):
        """
        Reset noise in noisy layers.
        """
        self.noisy_value.reset_noise()
        self.noisy_advantage.reset_noise()

    def get_q_values(self, q_atoms):
        """
        Compute Q-values as the expectation over the distribution.
        Args:
            q_atoms (torch.Tensor): Distributional Q-values for each action.
        Returns:
            torch.Tensor: Expected Q-values for each action.
        """
        return torch.sum(q_atoms * self.support, dim=2)


# Define global variables for model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = (6, 7)  # For a 6x7 Connect Four board
output_dim = 7     # Number of actions (columns)

# Load the trained DQN model
model_path = "./model/Rainbow3.pth"
dqn_model = RainbowDQN(input_dim, output_dim).to(device)
dqn_model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
dqn_model.eval()  # Set to evaluation mode

'''
observation: numpy.ndarray
shape. 6,7
0 for blank, 1 for mine, -1 for opponent

best_action: int
0-6 for next action
'''
def policy1(observation):
    observation = convert_state(observation.copy())
    
    # Flatten the observation and convert to tensor
    state = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension

    # Get Q-values for all actions
    with torch.no_grad():
        q_atoms = dqn_model(state)
        q_values = dqn_model.get_q_values(q_atoms)

    # Identify available actions (columns that are not full)
    available_actions = [col for col in range(q_values.shape[1]) if observation[0][col] == 0]
    
    # If all columns are full, return a random action (edge case)
    if not available_actions:
        return random.choice(range(q_values.shape[1]))

    # Choose the action with the highest Q-value among available actions
    best_action = max(available_actions, key=lambda col: q_values[0, col].item())
    
    return best_action

def policy2(observation):
    observation = convert_state(observation.copy())
    
    # Flatten the observation and convert to tensor
    state = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension

    # Get Q-values for all actions
    with torch.no_grad():
        q_atoms = dqn_model(state)
        q_values = dqn_model.get_q_values(q_atoms)

    # Identify available actions (columns that are not full)
    available_actions = [col for col in range(q_values.shape[1]) if observation[0][col] == 0]
    
    # If all columns are full, return a random action (edge case)
    if not available_actions:
        return random.choice(range(q_values.shape[1]))

    # Choose the action with the highest Q-value among available actions
    best_action = max(available_actions, key=lambda col: q_values[0, col].item())
    
    return best_action


# Convert 2 to -1 for better learning
def convert_state(state):
    state[state == 2] = -1
    return state

def flip_board(state):
    """Flip the board so that player 2 is treated as player 1 and vice versa."""
    state = state.copy()
    return state * -1



if __name__ == '__main__':
    from assets.connectfour import *
    env = ConnectFourEnv(rows=6, cols=7, render_mode= "human")
    # play_match(env, player1.policy1, player1.policy2)
    play_match(env, policy1, "human")
    # play_match(env, "human", player1.policy2)