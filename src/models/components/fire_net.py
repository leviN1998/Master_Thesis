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
        head: nn.Module = None,
        pretrained_weights: str = None,
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

        self.conv_flatten = nn.Conv2d(10, out_channels=16, kernel_size=1)

        if pretrained_weights is not None:
            print(f"Loading pretrained weights from {pretrained_weights}")
            checkpoint = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
            model_weights = checkpoint['state_dict']
            model_weights = {k.replace("net.", ""): v for k, v in model_weights.items() if "head." not in k}
            self.load_state_dict(model_weights, strict=False)
            print("Pretrained weights loaded (except head).")


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
        new_state1 = self.convgru1(x, self.state1)

        self.state1 = new_state1 * mask + self.state1 * (1 - mask)  # Apply mask to state
        x = self.state1

        x += residual
        # x = torch.relu(x)
        x = self.res1(x)

        residual = x
        new_state2 = self.convgru2(x, self.state2)
        self.state2 = new_state2 * mask + self.state2 * (1 - mask)  # Apply mask to state
        x = self.state2
        
        x += residual
        # x = torch.relu(x)

        return x


    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """ Forward pass through the FireNet model.
        
            args:
                x: Input tensor of shape (sequence_length, batch_size, input_channels, height, width).
                lengths: Optional tensor of lengths for each sequence in the batch.

            returns:
                torch.Tensor: Output tensor of shape (batch_size, hidden_channels, height, width).
        """
        #print("x shape before permute:", x.shape)
        x = x.permute(1, 0, 2, 3, 4)
        # x: [sequence_length, batch_size, input_channels, height, width]
        sequence_length, batch_size, input_channels, h, w = x.size()
        #print(sequence_length)
       
        # noise_frames = torch.randn(2, batch_size, input_channels, h, w, dtype=x.dtype, device=x.device)
        # x = torch.cat([x, noise_frames], dim=0)
        # sequence_length += 2

        self.state1 = torch.zeros(batch_size, self.hidden_channels, h, w, dtype=x.dtype, device=x.device)
        self.state2 = torch.zeros(batch_size, self.hidden_channels, h, w, dtype=x.dtype, device=x.device)

        output = torch.zeros_like(self.head(torch.zeros(batch_size, self.hidden_channels, h, w, dtype=x.dtype, device=x.device)))
        
        if lengths is None:
            lengths = torch.full((batch_size,), sequence_length, dtype=torch.int64, device=x.device)

        last_feat = torch.zeros(batch_size, self.hidden_channels, h, w, dtype=x.dtype, device=x.device)

        for t in range(sequence_length):
            continue
            mask = (lengths > t).float() # [batch]
            mask = mask.view(-1, 1, 1, 1)  # [batch, 1, 1, 1]
            x_t = x[t]

            x_t = self.forward_step(x_t, mask)

            for b in range(batch_size):
                if t == lengths[b] - 1:
                    last_feat[b] = x_t[b]

            """
            is_last_step = (lengths == (t + 1)).float().view(batch_size, 1, 1)
            x_t = self.res2(x_t)
            x_t = self.head(x_t)

            # print(torch.ones(x_t.shape[1:]))
            # print(f"x_t shape: {x_t.shape}, is_last_step shape: {is_last_step.shape}, output shape: {output.shape}")
            if len(is_last_step.shape[1:]) != len(output.shape[1:]):
                #print(is_last_step.shape, output.shape)
                is_last_step = is_last_step[:, None]

            output += x_t * is_last_step
            """

        #x = self.res2(last_feat)

        x = self.conv_flatten(x[5])
        output = self.head(x)

        return output