import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic import BaseModel
from typing import List


class EncoderConfig(BaseModel):
    num_layers: int = 2
    num_resnet_blocks: int = 2
    input_channels: int = 3
    output_channels: int = 512


class DecoderConfig(BaseModel):
    num_layers: int = 2
    num_resnet_blocks: int = 2
    input_channels: int = 512
    output_channels: int = 3


class KLSchedConfig(BaseModel):
    start: float = 0.0
    end: float = 1e-4
    warmup: float = 0.1
    cooldown: float = 0.2


class TempSchedConfig(BaseModel):
    start: float = 0.9
    end: float = 0.00625
    warmup: float = 0.0
    cooldown: float = 0.2


class DVAEConfig(BaseModel):
    image_size: int = 224
    hidden_size: int = 64
    encoder: EncoderConfig
    decoder: DecoderConfig
    num_embeddings: int = 64
    embedding_dim: int = 64
    temperature: float = 0.9
    learning_rate: float = 1e-3
    learning_rate_scheduler_min: float = 1e-2
    kl_div_scheduler: KLSchedConfig
    temperature_scheduler: TempSchedConfig
    batch_size: int = 512
    max_epochs: int = 50


class ResidualBlock(torch.nn.Module):
    """A Conv2D block with skip connections.

    A single ResidualBlock module computes the following:
        `y = relu(x + norm(conv(relu(norm(conv(x))))))`
    where `x` is the input, `y` is the output, `norm` is a 2D batch norm and `conv` is
    a 2D convolution with kernel of size 3, stride 1 and padding 1.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Init the ResidualBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.norm2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the ResidualBlock.

        Args:
            x: The input.

        Returns:
            The output after applying the residual block. See the class description
            for more details.
        """
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return self.relu(y + x)


class Conv2dEncoder(nn.Module):
    def __init__(self, config: DVAEConfig):
        super(Conv2dEncoder, self).__init__()
        self.layers_list = [
            nn.Conv2d(config.encoder.input_channels, config.hidden_size, kernel_size=1)
        ]

        for i in range(config.encoder.num_layers):
            self.layers_list.extend(
                [
                    ResidualBlock(config.hidden_size, config.hidden_size)
                    for _ in range(config.encoder.num_resnet_blocks)
                ]
            )
            self.layers_list.append(
                nn.Conv2d(
                    config.hidden_size,
                    config.hidden_size,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            self.layers_list.append(nn.ReLU())

        self.layers_list.append(
            nn.Conv2d(config.hidden_size, config.encoder.output_channels, kernel_size=1)
        )
        self.layers = nn.Sequential(*self.layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Conv2DDecoder(torch.nn.Module):
    """An image decoder based on 2D transposed convolutions.

    Based on: https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
    """

    def __init__(self, config: DVAEConfig):
        """Init the encoder.

        Args:
            input_channels: Number of input channels (dimensionality of the codebook).
            output_channels: Number of output channels.
            hidden_size: Number of channels in the intermediate hidden layers.
            num_layers: Number of hidden layers.
            num_resnet_blocks: Number of resnet blocks added after each layer.
        """
        super().__init__()
        layers_list: list[torch.nn.Module] = [
            torch.nn.Conv2d(
                config.decoder.input_channels, config.hidden_size, kernel_size=1
            )
        ]
        for _ in range(config.decoder.num_layers):
            layers_list.extend(
                [
                    ResidualBlock(config.hidden_size, config.hidden_size)
                    for _ in range(config.decoder.num_resnet_blocks)
                ]
            )
            layers_list.append(
                torch.nn.ConvTranspose2d(
                    config.hidden_size,
                    config.hidden_size,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            layers_list.append(torch.nn.ReLU())

        layers_list.append(
            torch.nn.Conv2d(
                config.hidden_size, config.decoder.output_channels, kernel_size=1
            )
        )
        layers_list.append(torch.nn.Sigmoid())
        self.layers = torch.nn.Sequential(*layers_list)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Encode an image given latent variables.

        Args:
            z: The input image of shape `(batch, input_channels, in_width, in_height)`.

        Returns:
            The encoder image of shape `(batch, output_channels, out_width, out_height)`.
        """
        return self.layers(z)


class DVAE(nn.Module):
    def __init__(self, config: DVAEConfig):
        super(DVAE, self).__init__()
        self.encoder = Conv2dEncoder(config=config)
        self.decoder = Conv2DDecoder(config=config)

        self.embeddings = nn.Embedding(config.num_embeddings, config.embedding_dim)
        nn.init.xavier_normal(self.embeddings.weight)
        self.temperature = config.temperature

    def encode(
        self, x: torch.Tensor, hard: bool = False, temperature: float | None = None
    ):

        logits = self.encoder(x).permute((0, 2, 3, 1))
        if temperature is None:
            temperature = self.temperature

        soft_one_hot = F.gumbel_softmax(logits, dim=-1, tau=temperature, hard=True)
        x_encoded = torch.matmul(
            soft_one_hot.unsqueeze(-2), self.embeddings.weight
        ).squeeze(-2)

        return x_encoded, logits

    def decode(self, z: torch.Tensor, from_indices: bool = False):
        if from_indices:
            z = self.embeddings(z)

        return self.decoder(z.permute((0, 3, 1, 2)))

    def forward(self, x: torch.Tensor, temperature: float | None = None):
        if temperature is None:
            temperature = self.temperature

        return self.encode(x, temperature=temperature)


if __name__ == "__main__":

    global_configs = DVAEConfig(
        image_size=224,
        hidden_size=64,
        encoder=EncoderConfig(
            input_channels=3, output_channels=512, num_layers=2, num_resnet_blocks=2
        ),
        decoder=DecoderConfig(
            num_layers=2,
            num_resnet_blocks=2,
            input_channels=512,
            output_channels=3,
        ),
    )
    print(global_configs)
    image = torch.randn(1, 3, 224, 224)

    model = DVAE(config=global_configs)
    print(model)
    # enc = Conv2dEncoder(global_configs)
    # dec = Conv2DDecoder(global_configs)
    # z = enc(image)
    # print(z.shape)
    # x = dec(z)
    # print(x.shape)
