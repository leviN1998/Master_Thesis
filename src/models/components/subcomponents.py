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
    

class AdaptiveAvgHead(nn.Module):
    """ Head with adaptive average pooling and a fully connected layer.
    """

    def __init__(self, input_channels: int, hidden_channels: int, num_classes: int, pretrained_weights: str = None):
        """ Initialize the adaptive average pooling head.
        
        :param input_channels: Number of input channels.
        :param num_classes: Number of output classes.
        """
        super().__init__()
        self.conv = nn.Conv2d(input_channels, out_channels=hidden_channels, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels, num_classes)
        if pretrained_weights is not None:
            print(f"Loading pretrained weights from {pretrained_weights}")
            checkpoint = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
            model_weights = checkpoint['state_dict']
            model_weights = {k.replace("net.head.", ""): v for k, v in model_weights.items() if "head." in k}
            self.load_state_dict(model_weights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the adaptive average pooling head.
        
        :param x: Input tensor of shape (batch_size, input_channels, height, width).
        :return: Output tensor of shape (batch_size, num_classes).
        """
        x = self.conv(x)
        x = torch.relu(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(x)
        x = self.fc(x)

        return x
    

class RegressionHead(nn.Module):
    """ Head for regression tasks.
    """

    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, pretrained_weights: str = None):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        
        self.conv = nn.Conv2d(input_channels, out_channels=1, kernel_size=1)
        self.fc = nn.Linear(100*100, output_channels)


        # TODO: Load pretrained weights if provided
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the regression head.
        
        :param x: Input tensor of shape (batch_size, input_channels, height, width).
        :return: Output tensor of shape (batch_size, output_channels).
        """
        x = self.conv(x)
        x = torch.relu(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)

        #M = x.view(-1, 3, 3)  
        M = x

        return M
    
class SmallHead(nn.Module):
    """ Small head for regression tasks.
    """

    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, dropout: float = 0.25):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.norm = nn.LayerNorm(input_channels)
        self.fc1 = nn.Linear(input_channels, hidden_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_channels, output_channels)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the regression head.
        
        :param x: Input tensor of shape (batch_size, input_channels, height, width).
        :return: Output tensor of shape (batch_size, output_channels).
        """
        x = self.pool(x)
        x = self.flatten(x)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc_out(x)

        M = x

        return M
    

class EasyNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(10000, 50)
        self.fc2 = nn.Linear(50, 9)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2, 3, 4)
        
        sequence_length, batch_size, input_channels, h, w = x.size()

        batch = x[5, :, 7, :, :]
        # print("batch shape:", batch.shape)
        x = batch.flatten(start_dim=1)
        # print("x shape:", x.shape)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x.view(-1, 3, 3)
    


class DebugNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_channels = 16
        hidden_channels = 32
        output_channels = 9
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.conv_flatten = nn.Conv2d(10, input_channels, kernel_size=1)
        self.conv = nn.Conv2d(input_channels, out_channels=hidden_channels, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels, output_channels)

        self.register_buffer("bias_identity", torch.eye(3).flatten())

        # TODO: Load pretrained weights if provided
        

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the regression head.
        
        :param x: Input tensor of shape (batch_size, input_channels, height, width).
        :return: Output tensor of shape (batch_size, output_channels).
        """
        x = x.permute(1, 0, 2, 3, 4)
        
        sequence_length, batch_size, input_channels, h, w = x.size()
        batch = x[5]

        x = self.conv_flatten(batch)
        x = self.conv(x)
        x = torch.relu(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)

        x = x + 0.01 * self.bias_identity
        M = x.view(-1, 3, 3)  

        return M