from torch import nn


class ResModule(nn.Module):
    """
    ReLU, 3x3 conv, ReLU, 1x1 conv with 256 hidden units.
    """

    def __init__(self, channels, hidden_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(x)
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        return x + h


def sequential_encoder(input_channels: int, output_channels: int, hidden_units=256):
    """
    The encoder consists of 2 strided convolutional layers with stride 2 and window size 4 × 4, followed by two residual
    3 × 3 blocks (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units.
    @param input_channels:
    @param output_channels:
    @param hidden_units:
    @return: the encoder module
    """

    return nn.Sequential(
        nn.Conv2d(input_channels, hidden_units, 4, stride=2),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.Conv2d(hidden_units, hidden_units, 4, stride=2),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        ResModule(hidden_units, hidden_units),
        ResModule(hidden_units, hidden_units),
        nn.Conv2d(hidden_units, output_channels, 1)
    )

def sequential_decoder(input_channels: int, output_channels: int, hidden_units=256):
    """
    The decoder consists of two residual 3 × 3 blocks, followed by two strided convolutional layers with stride 2 and
    window size 4 × 4, all having 256 hidden units.
    @param input_channels:
    @param output_channels:
    @param hidden_units:
    @return: the decoder module
    """

    return nn.Sequential(
        ResModule(input_channels),
        ResModule(input_channels),
        nn.ConvTranspose2d(input_channels, hidden_units, 4, stride=2, output_padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.ConvTranspose2d(hidden_units, hidden_units, 4, stride=2),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.Conv2d(hidden_units, output_channels, 1)
    )