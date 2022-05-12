from operator import concat
import timm.models.layers.activations
import torch
from torch import nn
from timm.models.convnext import LayerNorm2d
from functools import partial
import torch.nn.functional as F
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
    def __init__(self, spatial_shape, non_spatial_inputs, actions, activation=nn.ReLU, block=ConvNeXtBlock):
        super(MimicBotXNet, self).__init__()
        self.spatial_shape = spatial_shape
        self.non_spatial_shape = non_spatial_inputs
        self.actions = actions
        self.activation = activation
        self._create_spatial_processing()
        self._create_non_spatial_processing()
        self._create_combined_fc()
        self._create_attention()
        self._create_actor()
        self._create_critic()

    def _create_spatial_processing(self):
        self.spatial_processing = nn.Sequential(
            nn.Conv2d(44, 128, kernel_size=3, stride=1, padding=1),
            ConvNeXtBlock(dim=128),
            ConvNeXtBlock(dim=128),
            ConvNeXtBlock(dim=128),
            ConvNeXtBlock(dim=128)
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

        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1024, 64)
        self.act1 = self.activation(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Linear(1024, 64)
        self.act2 = self.activation(inplace=True)

        self.conv3 = nn.Conv2d(64, 17, kernel_size=3, stride=1, padding=1)
        self.fc3 = nn.Linear(1024, 17)
        self.act3 = self.activation(inplace=True)

    def _create_actor(self):
        self.actor_fc = nn.Linear(self.spatial_shape[1] * self.spatial_shape[2] * 17 + 1024, self.actions)

    def _create_critic(self):
        self.critic_fc = nn.Linear(1024, 1)

    def _spatial_processing(self, spatial_input) -> torch.Tensor:
        return self.spatial_processing(spatial_input)

    def _non_spatial_processing(self, non_spatial_input) -> torch.Tensor:
        return self.non_spatial_processing(non_spatial_input)

    def _combined_fc(self, combined) -> torch.Tensor:
        return self.combined_fc(combined)

    def _attention(self, spatial, non_spatial) -> torch.Tensor:
        x = self.conv1(spatial)
        y = self.fc1(non_spatial)
        y = self.act1(y)
        y = self.sigmoid(y)
        x = x * y.unsqueeze(-1).unsqueeze(-1)

        x = self.conv2(x)
        y = self.fc2(non_spatial)
        y = self.act2(y)
        y = self.sigmoid(y)

        x = x * y.unsqueeze(-1).unsqueeze(-1)

        x = self.conv3(x)
        y = self.fc3(non_spatial)
        y = self.act3(y)
        y = self.sigmoid(y)

        x = x * y.unsqueeze(-1).unsqueeze(-1)
        return x

    def _actor(self, spatial_actor, non_spatial_actor) -> torch.Tensor:
        spatial_actor = spatial_actor.flatten(start_dim=1)
        x = torch.concat([spatial_actor, non_spatial_actor], 1)
        return self.actor_fc(x)

    def _critic(self, combined) -> torch.Tensor:
        return self.critic_fc(combined)

    def forward(self, spatial_input: torch.Tensor, non_spatial_input: torch.Tensor):
        spatial_processed = self._spatial_processing(spatial_input)
        non_spatial_processed = self._non_spatial_processing(non_spatial_input)

        spatial_flattened = spatial_processed.flatten(start_dim=1)
        non_spatial_processed = non_spatial_processed.flatten(start_dim=1)

        concated = torch.cat([spatial_flattened, non_spatial_processed], 1)

        combined_out = self._combined_fc(concated)
        attention_out = self._attention(spatial_processed, combined_out)

        actor = self._actor(attention_out, combined_out)
        critic = self._critic(combined_out)

        return critic, actor

    @torch.jit.export
    def act(self, spatial_inputs, non_spatial_input, action_mask):
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

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask, scripted):
        value, policy = self(spatial_inputs, non_spatial_input)
        actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
        policy[~actions_mask] = float('-inf')
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        log_probs = torch.where(log_probs[None, :] == float('-inf') or scripted, torch.tensor(0.), log_probs)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy
