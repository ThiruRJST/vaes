import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, config):
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

    def __init__(self, config):
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


class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        self.config = config
        self.encoder = Conv2dEncoder(config=config)
        self.decoder = Conv2DDecoder(config=config)

        self.codebook = nn.Embedding(config.num_embeddings, config.embedding_dim)
        nn.init.xavier_normal(self.codebook.weight)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def vector_quantize(self, z):
        B, C, H, W = z.shape

        z_flattened = z.permute(0, 2, 3, 1).reshape(-1, C)
        distances = (
            (z_flattened.unsqueeze(1) - self.codebook.weight.unsqueeze(0))
            .pow(2)
            .sum(-1)
        )
        encoding_indices = distances.argmin(1)

        z_q = self.codebook(encoding_indices).view(B, H, W, C).permute(0, 3, 1, 2)
        z_q = z_q + (z - z_q).detach()

        return z_q, encoding_indices.view(B, H, W)

    def forward(self, x):
        z = self.encode(x)
        z_q, encoding_indices = self.vector_quantize(z)
        recon_x = self.decode(z_q)

        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z.detach(), z_q)
        loss = commitment_loss + codebook_loss

        p = (
            encoding_indices.view(-1)
            .bincount(minlength=self.config.num_embeddings)
            .float()
            + 1e-6
        )
        p /= p.sum()
        entropy_loss = -(p * p.log()).sum()
        
        loss = loss - entropy_loss

        return recon_x, loss, encoding_indices
