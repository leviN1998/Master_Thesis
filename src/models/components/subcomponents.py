import torch
from torch import nn

class ConvGRU(nn.Module):
    """ Convolutional Gated Recurrent Unit (ConvGRU) for processing sequences of images.
     
        Implementation of a ConvGRU Cell as described in the paper (DELVING DEEPER INTO CONVOLUTIONAL NETWORKS FOR LEARNING VIDEO REPRESENTATIONS)
        and used in Firenet.

        It consists of an update, reset, and output gate, but uses convolutions instead of multiplications.
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2 # to keep sizes
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.update_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.reset_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)


    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the ConvGRU cell.
        
        """
        # batch_size, channels, height, width = x.size()

        combined = torch.cat([x, state], dim=1)  # Concatenate input and hidden state

        update = self.update_gate(combined)
        update = torch.sigmoid(update)

        reset = self.reset_gate(combined)
        reset = torch.sigmoid(reset)

        state_tilde = self.out_gate(torch.cat([x, state * reset], dim=1))
        state_tilde = torch.tanh(state_tilde)
        new_state = state * (1 - update) + state_tilde * update

        return new_state



class ResidualBlock(nn.Module):
    """ A residual Block for Firenet
    
        As used for the Firenet Paper and expalined in the paper (Deep Residual Learning for Image Recognition)

        input ------>
          |          |
        Conv2d       |
          |          |
        ReLu         |
          |          |
        Conv2d       |
          |          |
          + <--------|
          |
        ReLu
          |
        Output

        Note: if different input and output chnannels desired this class needs to be modified
    """

    def __init__(self, input_channels: int, kernel_size: int = 3):
        super().__init__()
        self.input_channels = input_channels
        padding = kernel_size // 2  # to keep sizes

        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the Residual Block.
        
        """
        residual = x  # Save the input for the residual connection
        
        out = self.conv1(x)  
        out = self.relu(out)  
        out = self.conv2(out)  

        out += residual  
        out = self.relu(out)  
        return out 
    

class ClassificationHead(nn.Module):
    """ Head for the NMNIST dataset classification.
    """

    def __init__(self, input_channels: int, input_shape: list, num_classes: int):
        """ Initialize the classification head.
        
        :param input_channels: Number of input channels.
        :param input_shape: Shape of the input tensor (height, width).
        :param num_classes: Number of output classes.
        """
        super().__init__()
        self.conv = nn.Conv2d(input_channels, out_channels=1, kernel_size=1)
        self.shape = input_shape
        self.num_classes = num_classes
        self.fc = nn.Linear(1 * input_shape[0] * input_shape[1], num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the classification head.
        
        :param x: Input tensor of shape (batch_size, input_channels, height, width).
        :return: Output tensor of shape (batch_size, num_classes).
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(x)
        x = self.fc(x)

        return x