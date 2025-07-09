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


    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the ConvGRU cell.
        
        """
        batch_size, _, height, width = x.size()
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_channels, height, width, dtype=x.dtype, device=x.device)

        combined = torch.cat([x, hidden], dim=1)  # Concatenate input and hidden state
        z_t = torch.sigmoid(self.update_gate(combined))  # Update gate
        r_t = torch.sigmoid(self.reset_gate(combined))  # Reset gate
        h_tilde = torch.tanh(self.out_gate(torch.cat([x, r_t * hidden], dim=1)))
        h_t = (1 - z_t) * hidden + z_t * h_tilde  # New hidden state
        return h_t
    


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

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the Residual Block.
        
        """
        residual = x  # Save the input for the residual connection
        
        out = self.conv1(x)  
        out = torch.relu(out)  
        out = self.conv2(out)  

        out += residual  
        out = torch.relu(out)  
        return out 