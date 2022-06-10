from operator import concat
import timm.models.layers.activations
import torch
from torch import nn
from timm.models.convnext import LayerNorm2d
from functools import partial
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, ClassifierHead, SelectAdaptivePool2d, DropPath, \
                               ConvMlp, Mlp, create_conv2d, SqueezeExcite
from typing import Optional, Callable
from torch.utils.checkpoint import checkpoint

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetBlock(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        dim: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation=timm.models.layers.activations.GELU
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = dim * 3
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(dim, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1):
        super(SeparableConv2d, self).__init__()

        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=0, bias=bias)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x


class NEXcepTionBlock(nn.Module):
    def __init__(self, dim, strides=1, drop_path=0, activation=timm.models.layers.activations.GELU):
        super(NEXcepTionBlock, self).__init__()


        self.sepconv1 = SeparableConv2d(dim, dim * 3, kernel_size=5, stride=strides, padding=2)
        # if 1 in normaliztion_pos:
        self.norm1 = nn.BatchNorm2d(dim * 3)


        # if 2 in activation_pos:
        self.act = nn.GELU()

        self.sepconv2 = SeparableConv2d(dim * 3, dim, kernel_size=5, stride=strides, padding=2)
        # if 2 in normaliztion_pos:
        self.norm2 = nn.BatchNorm2d(dim)


        self.sepconv3 = SeparableConv2d(dim, dim, kernel_size=5, stride=strides, padding=2)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.block_end = SqueezeExcite(dim, act_layer=activation, norm_layer=nn.BatchNorm2d)
    
    def forward(self, x):
        skip = x
        x = self.sepconv1(x)
        x = self.norm1(x)
        x = self.sepconv2(x)
        x = self.act(x)
        x = self.sepconv3(x)
        x = self.block_end(x)
        x += skip

        return x


class MimicBotXNet(nn.Module):
    def __init__(self, spatial_shape, non_spatial_inputs, actions, activation=nn.ReLU, block=ConvNeXtBlock,
                    drop_path=0.0, drop_rate=0.0):
        super(MimicBotXNet, self).__init__()
        self.spatial_shape = spatial_shape
        self.non_spatial_shape = non_spatial_inputs
        self.actions = actions
        self.activation = activation
        self._create_spatial_processing(drop_path, block)
        self._create_non_spatial_processing()
        self._create_combined_fc()
        self._create_attention()
        self._create_actor(drop_rate)
        self._create_critic(drop_rate)

    def _create_spatial_processing(self, drop_path, block):
        self.spatial_processing = nn.Sequential(
            nn.Conv2d(44, 128, kernel_size=3, stride=1, padding=1),
            block(dim=128, activation=self.activation),
            block(dim=128, activation=self.activation),
            block(dim=128, activation=self.activation),
            block(dim=128, activation=self.activation)
        )

    def _create_non_spatial_processing(self):
        self.non_spatial_processing = nn.Sequential(
            nn.Linear(self.non_spatial_shape, 1024),
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
            nn.Linear(self.spatial_shape[1] * self.spatial_shape[2] * 128 + 1024, 1024),
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

        self.conv1 = nn.Conv2d(128 + 44, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1024, 64)
        self.norm1 = nn.BatchNorm2d(64)
        self.act1 = self.activation(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Linear(1024, 64)
        self.norm2 = nn.BatchNorm2d(64)
        self.act2 = self.activation(inplace=True)

        self.conv3 = nn.Conv2d(64, 17, kernel_size=3, stride=1, padding=1)
        self.fc3 = nn.Linear(1024, 17)
        self.norm3 = nn.BatchNorm2d(17)
        self.act3 = self.activation(inplace=True)

    def _create_actor(self, drop_rate):
        self.actor_dropout = nn.Dropout(drop_rate, inplace=True)
        self.actor_fc = nn.Linear(self.spatial_shape[1] * self.spatial_shape[2] * 17 + 1024, self.actions)

    def _create_critic(self, drop_rate):
        self.critic_dropout = nn.Dropout(drop_rate, inplace=True)
        self.critic_fc = nn.Linear(self.spatial_shape[1] * self.spatial_shape[2] * 17 + 1024, 1)

    def _spatial_processing(self, spatial_input) -> torch.Tensor:
        return self.spatial_processing(spatial_input)

    def _non_spatial_processing(self, non_spatial_input) -> torch.Tensor:
        return self.non_spatial_processing(non_spatial_input)

    def _combined_fc(self, combined) -> torch.Tensor:
        return self.combined_fc(combined)

    def _attention(self, spatial, non_spatial) -> torch.Tensor:
        x = self.conv1(spatial)
        x = self.norm1(x)
        y = self.fc1(non_spatial)
        y = self.act1(y)
        y = self.sigmoid(y)
        x = x * y.unsqueeze(-1).unsqueeze(-1)

        x = self.conv2(x)
        x = self.norm2(x)
        y = self.fc2(non_spatial)
        y = self.act2(y)
        y = self.sigmoid(y)

        x = x * y.unsqueeze(-1).unsqueeze(-1)

        x = self.conv3(x)
        x = self.norm3(x)
        y = self.fc3(non_spatial)
        y = self.act3(y)
        y = self.sigmoid(y)

        x = x * y.unsqueeze(-1).unsqueeze(-1)
        return x

    def _actor(self, spatial_actor, non_spatial_actor) -> torch.Tensor:
        spatial_actor = spatial_actor.flatten(start_dim=1)
        x = torch.concat([spatial_actor, non_spatial_actor], 1)
        return self.actor_fc(x)

    def _critic(self, spatial_actor, non_spatial_actor) -> torch.Tensor:
        spatial_actor = spatial_actor.flatten(start_dim=1)
        x = torch.concat([spatial_actor, non_spatial_actor], 1)
        return self.critic_fc(x)

    def forward(self, spatial_input: torch.Tensor, non_spatial_input: torch.Tensor):
        # spatial_processed = self._spatial_processing(spatial_input)
        spatial_processed = self._spatial_processing(spatial_input)
        non_spatial_processed = self._non_spatial_processing(non_spatial_input)

        spatial_flattened = spatial_processed.flatten(start_dim=1)
        non_spatial_processed = non_spatial_processed.flatten(start_dim=1)

        concated = torch.cat([spatial_flattened, non_spatial_processed], 1)

        combined_out = self._combined_fc(concated)
        attention_out = self._attention(torch.cat([spatial_processed, spatial_input], 1), combined_out)

        actor = self._actor(attention_out, combined_out)
        critic = self._critic(attention_out, combined_out)

        return critic, actor

    @torch.jit.export
    def act(self, spatial_inputs, non_spatial_input, action_mask):
        # print(spatial_inputs.size(), non_spatial_input.size(), action_mask.size())

        values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
        actions = action_probs.multinomial(1)
        return values, actions

    @torch.jit.export
    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            actions[~action_mask] = float('-inf')
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        value, policy = self(spatial_inputs, non_spatial_input)
        actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
        policy[~actions_mask] = float('-inf')
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        log_probs = torch.where(log_probs[None, :] == float('-inf'), torch.tensor(0., device='cuda:0'), log_probs)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy
