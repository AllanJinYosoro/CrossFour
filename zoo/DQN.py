import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(DQN, self).__init__()
        
        # Store input shape as an instance variable
        self.input_shape = input_shape
        
        # Convolutional layers to process the board
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        conv_output_size = 128 * input_shape[0] * input_shape[1]  # Flattened size after conv layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        # Reshape input to (batch_size, 1, rows, cols) for CNN
        x = x.view(-1, 1, *self.input_shape)  # Using self.input_shape to access rows and cols
        
        # Pass through convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the convolutional output
        x = torch.relu(self.fc1(x))
        return self.fc2(x)