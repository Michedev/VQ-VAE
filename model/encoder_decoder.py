from typing import List

from torch import nn


class ResModule(nn.Module):
    """
    ReLU, 3x3 conv, ReLU, 1x1 conv with 256 hidden units.
    """

    def __init__(self, channels, hidden_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(1, hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, channels, 1)
        self.gn2 = nn.GroupNorm(1, channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(x)
        h = self.conv1(h)
        h = self.gn1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        return x + h


def sequential_encoder(input_channels: int, output_channels: int, hidden_channels=256,
                       downsample_blocks: int = 2, res_blocks: int = 2):
    """
    The encoder consists of 2 strided convolutional layers with stride 2 and window size 4 × 4, followed by two residual
    3 × 3 blocks (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units.
    @param input_channels:
    @param output_channels:
    @param hidden_channels:
    @return: the encoder module
    """
    assert isinstance(downsample_blocks, int) and downsample_blocks >= 1, f'{downsample_blocks=} must be >= 1'
    assert isinstance(res_blocks, int) and res_blocks >= 1, f'{res_blocks=} must be >= 1 - '
    encoder = []
    for _ in range(downsample_blocks):
        encoder = encoder + [
            nn.Conv2d(input_channels, hidden_channels, 4, stride=2),
            nn.GroupNorm(1, hidden_channels),
            nn.ReLU(),
        ]
        input_channels = hidden_channels
    for _ in range(res_blocks):
        encoder.append(ResModule(hidden_channels, hidden_channels))
    encoder.append(nn.Conv2d(hidden_channels, output_channels, 1))

    encoder = nn.Sequential(*encoder)
    encoder.input_channels = input_channels
    encoder.output_channels = output_channels
    return encoder


def sequential_decoder(input_channels: int, output_channels: int, hidden_channels=256,
                       upsample_blocks: int = 2, res_blocks: int = 2):
    """
    The decoder consists of two residual 3 × 3 blocks, followed by two strided convolutional layers with stride 2 and
    window size 4 × 4, all having 256 hidden units.
    @param input_channels:
    @param output_channels:
    @param hidden_channels:
    @return: the decoder module
    """
    assert isinstance(upsample_blocks,
                      int) and upsample_blocks > 0, f'upsample_blocks must be >= 1 - {upsample_blocks=}'
    decoder: List[nn.Module] = []
    for _ in range(res_blocks):
        decoder.append(ResModule(input_channels, hidden_channels))
    for _ in range(upsample_blocks):
        decoder = decoder + [
            nn.ConvTranspose2d(input_channels, hidden_channels, 4, stride=2, output_padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.ReLU(),
        ]
        input_channels = hidden_channels
    decoder.append(nn.Conv2d(hidden_channels, output_channels, 1))

    decoder: nn.Module = nn.Sequential(*decoder)
    decoder.input_channels = input_channels
    decoder.output_channels = output_channels
    return decoder
