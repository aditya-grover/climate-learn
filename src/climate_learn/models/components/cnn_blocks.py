import torch
from torch import nn


class PeriodicPadding2D(nn.Module):
    def __init__(self, pad_width, **kwargs):
        super().__init__(**kwargs)
        self.pad_width = pad_width

    def forward(self, inputs, **kwargs):
        if self.pad_width == 0:
            return inputs
        inputs_padded = torch.cat(
            (
                inputs[:, :, :, -self.pad_width :],
                inputs,
                inputs[:, :, :, : self.pad_width],
            ),
            dim=-1,
        )
        # Zero padding in the lat direction
        inputs_padded = nn.functional.pad(
            inputs_padded, (0, 0, self.pad_width, self.pad_width)
        )
        return inputs_padded


class PeriodicConv2D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs
    ):
        super().__init__(**kwargs)
        self.padding = PeriodicPadding2D(padding)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, inputs):
        return self.conv(self.padding(inputs))


class PeriodicConvTranspose2D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs
    ):
        super().__init__(**kwargs)
        self.padding = PeriodicPadding2D(padding)
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, inputs):
        return self.conv(self.padding(inputs))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "leaky",
        norm: bool = False,
        dropout: float = 0.1,
        n_groups: int = 1,
    ):
        super().__init__()
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.3)
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.conv1 = PeriodicConv2D(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = PeriodicConv2D(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # First convolution layer
        # h = self.drop(self.conv1(self.activation(self.norm1(x))))
        h = self.drop(self.norm1(self.activation(self.conv1(x))))
        # Second convolution layer
        # h = self.drop(self.conv2(self.activation(self.norm2(h))))
        h = self.drop(self.norm2(self.activation(self.conv2(h))))
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """### Attention block This is similar to [transformer multi-head
    attention](../../transformers/mha.html)."""

    def __init__(
        self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 1
    ):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.BatchNorm2d(n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class DownBlock(nn.Module):
    """### Down block This combines `ResidualBlock` and `AttentionBlock`.

    These are used in the first half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "leaky",
        norm: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels,
            out_channels,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """### Up block This combines `ResidualBlock` and `AttentionBlock`.

    These are used in the second half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "leaky",
        norm: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels,
            out_channels,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """### Middle block It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(
        self,
        n_channels: int,
        has_attn: bool = False,
        activation: str = "leaky",
        norm: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels,
            n_channels,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(
            n_channels,
            n_channels,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    """### Scale up the feature map by $2 \times$"""

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    """### Scale down the feature map by $\frac{1}{2} \times$"""

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)
