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