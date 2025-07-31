""" Implementation of the fire net Model as described by the paper (Fast Image Reconstruction with an event camera)

    Architecture: (Components are defined in src/models/components/subcomponents.py)

        imput: (n=5) channel event tensor
        Convolution + ReLU layer
        ConvGRU layer + ReLU
        Residual Block
        ConvGRU layer + ReLU
        Residual Block
        Head 
"""

import torch
from torch import nn
from src.models.components.subcomponents import ConvGRU, ResidualBlock, ClassificationHead

class FireNet(nn.Module):
    """ A FireNet model for event-based image reconstruction.

        The model consists of a series of ConvGRU cells and residual blocks to process event data.
    """

    def __init__(
        self, 
        input_channels: int = 5, 
        hidden_channels: int = 16, 
        kernel_size: int = 3,
        head: nn.Module = None
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  # to keep sizes
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.res1 = ResidualBlock(hidden_channels, kernel_size)
        self.res2 = ResidualBlock(hidden_channels, kernel_size)
        self.convgru1 = ConvGRU(hidden_channels, hidden_channels, kernel_size)
        self.convgru2 = ConvGRU(hidden_channels, hidden_channels, kernel_size)
        self.head = head
        if self.head is None:
            self.head = ClassificationHead(hidden_channels, (34, 34), num_classes=10)


        self.state1 = None
        self.state2 = None


    def forward_step(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Forward step for a single time step in the sequence.
        
            args:
                x: Input tensor of shape (batch_size, input_channels, height, width).

            returns:
                torch.Tensor: Output tensor after processing through the model. Only if its the last timestep
        """
        x = self.conv(x)
        x = torch.relu(x)

        residual = x
        x = self.convgru1(x, self.state1)

        self.state1 = x * mask + self.state1 * (1 - mask)  # Apply mask to state

        x += residual
        x = torch.relu(x)
        x = self.res1(x)

        residual = x
        x = self.convgru2(x, self.state2)
        self.state2 = x * mask + self.state2 * (1 - mask)  # Apply mask to state

        x += residual
        x = torch.relu(x)

        return x


    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """ Forward pass through the FireNet model.
        
            args:
                x: Input tensor of shape (sequence_length, batch_size, input_channels, height, width).
                lengths: Optional tensor of lengths for each sequence in the batch.

            returns:
                torch.Tensor: Output tensor of shape (batch_size, hidden_channels, height, width).
        """

        x = x.permute(1, 0, 2, 3, 4)
        # x: [sequence_length, batch_size, input_channels, height, width]
        sequence_length, batch_size, input_channels, h, w = x.size()
       
        # noise_frames = torch.randn(2, batch_size, input_channels, h, w, dtype=x.dtype, device=x.device)
        # x = torch.cat([x, noise_frames], dim=0)
        # sequence_length += 2

        self.state1 = torch.zeros(batch_size, self.hidden_channels, h, w, dtype=x.dtype, device=x.device)
        self.state2 = torch.zeros(batch_size, self.hidden_channels, h, w, dtype=x.dtype, device=x.device)

        output = torch.zeros_like(self.head(torch.zeros(batch_size, self.hidden_channels, h, w, dtype=x.dtype, device=x.device)))
        if lengths is None:
            lengths = torch.full((batch_size,), sequence_length, dtype=torch.int64, device=x.device)

        for t in range(sequence_length):
            mask = (lengths > t).float() # [batch]
            mask = mask.view(-1, 1, 1, 1)  # [batch, 1, 1, 1]
            x_t = x[t]

            x_t = self.forward_step(x_t, mask)

            is_last_step = (lengths == (t + 1)).float().unsqueeze(1)
            x_t = self.res2(x_t)
            x_t = self.head(x_t)

            output += x_t * is_last_step

        
        return x_t