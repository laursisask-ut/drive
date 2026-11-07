from torch import nn
import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models
from pufferlib.ocean.drive import observation as drive_obs

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401


Recurrent = pufferlib.models.LSTMWrapper


class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

        # Determine ego dimension from environment's dynamics model
        self.ego_dim = drive_obs.ego_features(env.dynamics_model)

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(14, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        max_partner_objects = 7
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(max_partner_objects, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self(x)

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = 63 * 7
        road_dim = 200 * 7
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, 63, 7)
        road_objects = road_obs.view(-1, 200, 7)
        road_continuous = road_objects[:, :, :6]  # First 6 features
        road_categorical = road_objects[:, :, 6]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=8)  # Type 11 (yield_right) is category 7.
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value

class SceneEncoderBlock(torch.nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm1 = torch.nn.LayerNorm(hidden_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 4 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.layer_norm2 = torch.nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        norm_x = self.layer_norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out

        mlp_out = self.mlp(self.layer_norm2(x))
        return x + mlp_out

class SceneEncoder(torch.nn.Module):
    def __init__(self, ego_dim=98, hidden_dim=256, num_blocks=4):
        super().__init__()
        self.ego_dim = ego_dim
        if ego_dim not in (98, 101):
            raise ValueError("Unsupported Drive ego layout; regenerate observations and retrain old checkpoints")
        # All pose, speed and selected-command history belongs to this one token.
        self.ego_state_indices = list(range(9, 15)) + list(range(21, ego_dim))
        self.goal_indices = [15, 16, 19, 20]
        self.next_goal_indices = [17, 18, 19, 20]
        self.reward_pref_indices = list(range(9))
        self.map_proj = torch.nn.Sequential(
            torch.nn.Linear(14, hidden_dim),
            torch.nn.ReLU()
        )
        self.vehicle_proj = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.ReLU()
        )
        self.ego_state_proj = torch.nn.Sequential(
            torch.nn.Linear(len(self.ego_state_indices), hidden_dim),
            torch.nn.ReLU()
        )
        self.goal_proj = torch.nn.Sequential(
            torch.nn.Linear(len(self.goal_indices), hidden_dim),
            torch.nn.ReLU()
        )
        self.next_goal_proj = torch.nn.Sequential(
            torch.nn.Linear(len(self.next_goal_indices), hidden_dim),
            torch.nn.ReLU()
        )
        self.reward_pref_proj = torch.nn.Sequential(
            torch.nn.Linear(len(self.reward_pref_indices), hidden_dim),
            torch.nn.ReLU()
        )
        self.reward_film = torch.nn.Sequential(
            torch.nn.Linear(len(self.reward_pref_indices), hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        self.blocks = torch.nn.ModuleList([SceneEncoderBlock(hidden_dim=hidden_dim) for _ in range(num_blocks)])
        self.final_norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, observations):
        B = observations.size(0)
        V = 63
        M = 200

        ego_dim = self.ego_dim
        partner_dim = V * 7
        road_dim = M * 7

        ego_data = observations[:, :ego_dim]
        vehicle_data = observations[:, ego_dim : ego_dim + partner_dim].view(-1, V, 7)
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim].view(-1, M, 7)

        road_continuous = road_obs[:, :, :6]
        road_categorical = road_obs[:, :, 6]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=8)
        map_data = torch.cat([road_continuous, road_onehot], dim=2)

        map_tokens = self.map_proj(map_data.view(B * M, -1)).view(B, M, -1)
        vehicle_tokens = self.vehicle_proj(vehicle_data.reshape(B * V, -1)).view(B, V, -1)
        ego_state_token = self.ego_state_proj(ego_data[:, self.ego_state_indices]).view(B, 1, -1)
        goal_token = self.goal_proj(ego_data[:, self.goal_indices]).view(B, 1, -1)
        next_goal_token = self.next_goal_proj(ego_data[:, self.next_goal_indices]).view(B, 1, -1)
        reward_prefs = ego_data[:, self.reward_pref_indices]
        reward_pref_token = self.reward_pref_proj(reward_prefs).view(B, 1, -1)
        ego_tokens = torch.cat([ego_state_token, goal_token, next_goal_token, reward_pref_token], dim=1)

        x = torch.cat([map_tokens, vehicle_tokens, ego_tokens], dim=1)
        gamma, beta = self.reward_film(reward_prefs).chunk(2, dim=-1)
        x = x * (1 + gamma[:, None, :]) + beta[:, None, :]

        for block in self.blocks:
            x = block(x)

        return self.final_norm(x)

class TrajectoryDecoderBlock(torch.nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm1 = torch.nn.LayerNorm(hidden_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 4 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.layer_norm2 = torch.nn.LayerNorm(hidden_dim)
    
    def forward(self, x, encoder_outputs):
        norm_x = self.layer_norm1(x)
        attn_out, _ = self.attn(norm_x, encoder_outputs, encoder_outputs, need_weights=False)
        x = x + attn_out

        mlp_out = self.mlp(self.layer_norm2(x))
        return x + mlp_out

class TrajectoryDecoder(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim=256, num_blocks=4):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seed = torch.nn.Parameter(torch.randn((1, 1, hidden_dim)) * 0.02)
        self.blocks = torch.nn.ModuleList([TrajectoryDecoderBlock(hidden_dim=hidden_dim) for _ in range(num_blocks)])
        self.final_norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.actor = pufferlib.pytorch.layer_init(torch.nn.Linear(hidden_dim, sum(action_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(torch.nn.Linear(hidden_dim, 1), std=1)
    
    def forward(self, encoder_outputs):
        x = self.seed.repeat(encoder_outputs.size(0), 1, 1)
        for block in self.blocks:
            x = block(x, encoder_outputs)

        x = self.final_norm(x)
        actions, value = self.decode_actions(x.reshape(-1, self.hidden_dim))
        return actions, value

    def decode_actions(self, x):
        action = self.actor(x)
        action = torch.split(action, self.action_dim, dim=1)
        value = self.value_fn(x)
        return action, value

class Wayformer(torch.nn.Module):
    def __init__(self, env=None, ego_dim=None, action_dim=None, **kwargs):
        super().__init__()
        hidden_dim = 128
        if ego_dim is None:
            ego_dim = drive_obs.ego_features(env.dynamics_model if env is not None else "classic")
        if action_dim is None:
            action_dim = env.single_action_space.nvec.tolist()
        self.encoder = SceneEncoder(ego_dim=ego_dim, hidden_dim=hidden_dim, num_blocks=2)
        self.decoder = TrajectoryDecoder(action_dim=action_dim, hidden_dim=hidden_dim, num_blocks=1)

    def forward(self, x, state=None):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_eval(self, observations, state=None):
        return self(observations)
