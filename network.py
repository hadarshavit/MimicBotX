import timm.models.layers.activations
import torch
from torch import nn
from timm.models.convnext import LayerNorm2d
from functools import partial
import torch.functional as F
from timm.models.layers import trunc_normal_, ClassifierHead, SelectAdaptivePool2d, DropPath, ConvMlp, Mlp

class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., ls_init_value=1e-6, conv_mlp=False, mlp_ratio=4, norm_layer=None,
                 activation=timm.models.layers.activations.GELU):
        super().__init__()
        if not norm_layer:
            norm_layer = partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.use_conv_mlp = conv_mlp
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=activation)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

class MimicBotXNet(nn.Module):
    def __init__(self, activation=nn.ReLU, block=ConvNeXtBlock):
        self.activation = activation

    def _create_spatial_processing(self):
        self.spatial_processing = nn.Sequential(
            nn.Conv2d(43, 128, kernel_size=3, stride=1, padding=1),
            ConvNeXtBlock(dim=128),
            ConvNeXtBlock(dim=128),
            ConvNeXtBlock(dim=128),
            ConvNeXtBlock(dim=128)
        )

    def _create_non_spatial_processing(self):
        self.non_spatial_processing = nn.Sequential(
            nn.Linear(116, 1024),
            self.activation(inplace=True),
            nn.Linear(1024, 1024),
            self.activation(inplace=True),
            nn.Linear(1024, 1024),
            self.activation(inplace=True),
            nn.Linear(1024, 1024),
            self.activation(inplace=True)
        )

    def _create_combined_fc(self):
        self.combined_fc = nn.Sequential(
            nn.Linear(61952, 1024),
            self.activation(inplace=True),
            nn.Linear(1024, 1024),
            self.activation(inplace=True),
            nn.Linear(1024, 1024),
            self.activation(inplace=True),
            nn.Linear(1024, 1024),
            self.activation(inplace=True)
        )

    def _create_attention(self):
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(1024, 64)
        self.act1 = self.activation(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc2 = nn.Linear(1024, 64)
        self.act2 = self.activation(inplace=True)

        self.conv3 = nn.Conv2d(64, 17, kernel_size=3, stride=1)
        self.fc3 = nn.Linear(1024, 17)
        self.act3 = self.activation(inplace=True)

    def _create_actor(self):
        pass

    def _create_critic(self):
        pass

    def _spatial_processing(self, spatial_input) -> torch.Tensor:
        pass

    def _non_spatial_processing(self, non_spatial_input) -> torch.Tensor:
        pass

    def _combined_fc(self, combined) -> torch.Tensor:
        pass

    def _attention(self, spatial, combined_fc) -> torch.Tensor:
        pass

    def _actor(self, spatial_actor, non_spatial_actor) -> torch.Tensor:
        pass

    def _critic(self, combined) -> torch.Tensor:
        pass

    def forward(self, spatial_input: torch.Tensor, non_spatial_input: torch.Tensor):
        spatial_processed = self._spatial_processing(spatial_input)
        non_spatial_processed = self._non_spatial_processing(non_spatial_input)

        spatial_flattened =  spatial_processed.flatten(start_dim=1)
        concated = torch.concat([spatial_flattened, non_spatial_processed])

        combined_out = self._combined_fc(concated)

        attention_out = self._attention(spatial_processed, combined_out)

        actor = self._actor(attention_out, combined_out)
        critic = self._critic(combined_out)

        return critic, actor
