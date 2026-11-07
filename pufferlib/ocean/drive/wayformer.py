"""Checkpoint-preserving execution of Wayformer's repeated padding tokens.

Padding is *not* masked in the original model. A run of n identical tokens can
instead be represented once, adding log(n) to its key's attention score in
every encoder and decoder layer. Layer norms and MLPs operate independently
on tokens, so this also preserves parameter gradients. No positional encoding
or dropout is present in Wayformer.
"""

import functools

import torch
from torch.nn import functional as F


CAPACITIES = (64, 96, 128, 160, 192, 224, 272)


def prepare_batch(observations, ego_dim):
    valid = observations[:, ego_dim : ego_dim + 263 * 7].reshape(-1, 263, 7).ne(0).any(-1)
    vehicles = valid[:, :63].sum(-1)
    maps = valid[:, 63:].sum(-1)
    vehicle_positions = torch.arange(63, device=observations.device)
    map_positions = torch.arange(200, device=observations.device)
    holes = (valid[:, :63] & (vehicle_positions >= vehicles[:, None])).any() | (
        valid[:, 63:] & (map_positions >= maps[:, None])
    ).any()
    # Four ego tokens plus one representative of each kind of padding.
    capacity = torch.tensor(CAPACITIES, device=observations.device)
    buckets = torch.searchsorted(capacity, maps + vehicles + 6)
    order = buckets.argsort()
    info = torch.stack([(buckets == i).sum() for i in range(len(CAPACITIES))] + [holes.long()])
    counts = torch.stack([maps, vehicles], -1)
    return order, info, counts


def forward_group(model, observations, counts, capacity, fast_attention=False):
    """Run a group with trailing padding; counts are (map, vehicle) per scene."""
    encoder, decoder = model.encoder, model.decoder
    batch = observations.size(0)
    width = decoder.hidden_dim
    ego_dim = encoder.ego_dim
    ego = observations[:, :ego_dim]
    maps, vehicles = counts[:, :1], counts[:, 1:]
    position = torch.arange(capacity - 4, device=observations.device)[None, :]
    # Original token ordering is 200 maps, 63 vehicles, then four ego tokens.
    indices = torch.where(
        position < maps,
        position,
        torch.where(
            position < maps + vehicles,
            200 + position - maps,
            torch.where(
                position == maps + vehicles,
                maps.clamp_max(199),
                torch.where(position == maps + vehicles + 1, 200 + vehicles.clamp_max(62), 0),
            ),
        ),
    )
    multiplicity = torch.where(
        position < maps + vehicles,
        1,
        torch.where(
            position == maps + vehicles,
            200 - maps,
            torch.where(position == maps + vehicles + 1, 63 - vehicles, 0),
        ),
    )
    multiplicity = torch.cat([multiplicity, torch.ones_like(counts[:, :1]).expand(-1, 4)], dim=1)
    bias_dtype = torch.float64 if observations.dtype == torch.float64 else torch.float32
    key_bias = multiplicity.to(bias_dtype).log()[:, None, None, :]

    is_map = indices < 200
    start = torch.where(is_map, ego_dim + 441 + 7 * indices, ego_dim + 7 * (indices - 200))
    features = start[:, :, None] + torch.arange(7, device=observations.device)
    raw = observations.gather(1, features.reshape(batch, -1)).reshape(batch, capacity - 4, 7)
    category = torch.where(is_map, raw[:, :, 6], 0).long()
    map_data = torch.cat([raw[:, :, :6], F.one_hot(category, 8)], dim=-1)

    # One linear operation with disjoint feature columns retains both original
    # projections and biases. This avoids computing both projections per token.
    map_projection, vehicle_projection = encoder.map_proj[0], encoder.vehicle_proj[0]
    object_data = torch.cat(
        [
            torch.where(is_map[:, :, None], map_data, 0),
            torch.where(is_map[:, :, None], 0, raw),
            is_map[:, :, None],
            (~is_map)[:, :, None],
        ],
        dim=-1,
    )
    projection_weight = torch.cat(
        [
            map_projection.weight,
            vehicle_projection.weight,
            map_projection.bias[:, None],
            vehicle_projection.bias[:, None],
        ],
        dim=-1,
    )
    objects = F.relu(F.linear(object_data, projection_weight))

    reward = ego[:, encoder.reward_pref_indices]
    ego_tokens = torch.stack(
        [
            encoder.ego_state_proj(ego[:, encoder.ego_state_indices]),
            encoder.goal_proj(ego[:, encoder.goal_indices]),
            encoder.next_goal_proj(ego[:, encoder.next_goal_indices]),
            encoder.reward_pref_proj(reward),
        ],
        dim=1,
    )
    x = torch.cat([objects, ego_tokens], dim=1)
    gamma, beta = encoder.reward_film(reward).chunk(2, dim=-1)
    x = x * (1 + gamma[:, None, :]) + beta[:, None, :]
    for block in encoder.blocks:
        x = x + model._attention(
            block.attn,
            block.layer_norm1(x),
            key_bias=key_bias,
            use_fast=fast_attention,
        )
        x = x + block.mlp(block.layer_norm2(x))
    encoded = encoder.final_norm(x)

    x = decoder.seed.expand(batch, -1, -1)
    for block in decoder.blocks:
        x = x + model._attention(
            block.attn,
            block.layer_norm1(x),
            encoded,
            key_bias=key_bias,
            use_fast=fast_attention,
        )
        x = x + block.mlp(block.layer_norm2(x))
    return decoder.decode_actions(decoder.final_norm(x).reshape(batch, width))


def _dense(model, observations):
    return model._forward_dense(observations)


@functools.lru_cache(maxsize=1)
def _compiled_functions():
    # These functions take the model explicitly: deepcopy and state_dict never
    # retain a compiled closure referring to a different model's parameters.
    return (
        torch.compile(prepare_batch, dynamic=True),
        torch.compile(_dense, dynamic=True),
    )


@functools.lru_cache(maxsize=3 * len(CAPACITIES))
def _compiled_group(capacity, mode):
    def run(model, observations, counts):
        return forward_group(model, observations, counts, capacity, True)

    # Dynamo caches by Python code object, not function instance. Give each
    # capacity/mode its own cache so small/large batch specializations cannot
    # exhaust the default eight-variant limit across all seven capacities.
    # Only code metadata/identity changes; the executable instructions are
    # identical. The closure retains a capacity, never model parameters.
    name = f"wayformer_{mode}_{capacity}"
    run.__code__ = run.__code__.replace(co_name=name)
    run.__name__ = run.__qualname__ = name
    return torch.compile(run, dynamic=True)


@torch.compiler.disable
def compact_forward(model, observations):
    prepare, dense = _compiled_functions()
    amp_bfloat16 = torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
    native_bfloat16 = model.encoder.map_proj[0].weight.dtype == torch.bfloat16
    blocks = [*model.encoder.blocks, *model.decoder.blocks]
    if (
        torch.cuda.get_device_capability(observations.device) != (10, 0)
        or observations.dtype != torch.bfloat16
        or not (amp_bfloat16 or native_bfloat16)
        or any(block.attn.dropout != 0 for block in blocks)
    ):
        return dense(model, observations)

    order, info, counts = prepare(observations, model.encoder.ego_dim)
    # One host transfer provides the seven group sizes and validates the
    # trailing-zero assumption. Unusual layouts retain the dense execution.
    *sizes, holes = info.tolist()
    if holes or sum(size * cap for size, cap in zip(sizes, CAPACITIES)) >= observations.size(0) * 267 * 0.85:
        return dense(model, observations)
    sorted_observations = observations[order]
    sorted_counts = counts[order]
    mode = "training" if torch.is_grad_enabled() else ("rollout" if model.decoder.seed.requires_grad else "anchor")
    actions, values = [], []
    start = 0
    for size, capacity in zip(sizes, CAPACITIES):
        if size:
            group_observations = sorted_observations[start : start + size]
            group_counts = sorted_counts[start : start + size]
            if size == 1:
                # Dynamo specializes singleton dimensions. One unused duplicate
                # keeps the group batch dimension dynamic without an extra
                # compiled variant; its output contributes no parameter gradient.
                group_observations = group_observations.repeat(2, 1)
                group_counts = group_counts.repeat(2, 1)
            logits, value = _compiled_group(capacity, mode)(
                model,
                group_observations,
                group_counts,
            )
            actions.append(torch.cat(logits, dim=-1)[:size])
            values.append(value[:size])
        start += size
    inverse = order.argsort()
    logits = torch.cat(actions)[inverse]
    value = torch.cat(values)[inverse]
    return logits.split(model.decoder.action_dim, dim=-1), value
