"""B200 BF16 attention for Wayformer's short, weighted token sequences.

CUDA kernels implement the existing scaled dot-product attention, including
first-order gradients. Traceable dispatcher operators let Inductor optimize
the surrounding tensor conversions and launches. Kernel variants depend on
token capacity, never on the number of scenes in a group.
"""

import torch
import triton as tr
import triton.language as tl
from torch.library import wrap_triton


@tr.jit
def decode_fwd(
    Q,
    K,
    V,
    Bias,
    O,
    Lse,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    q = tl.load(Q + b * QS0 + h * QS1 + d).to(tl.float32)
    k = tl.load(K + b * KS0 + h * KS1 + n[:, None] * KS2 + d[None, :], n[:, None] < N, 0).to(tl.float32)
    bias = tl.load(Bias + b * N + n, n < N, float("-inf"))
    scores = tl.sum(k * q[None, :], 1) * SCALE + bias
    maximum = tl.max(scores, 0)
    p = tl.exp(scores - maximum)
    denom = tl.sum(p, 0)
    p = p / denom
    v = tl.load(V + b * VS0 + h * VS1 + n[:, None] * VS2 + d[None, :], n[:, None] < N, 0).to(tl.float32)
    out = tl.sum(p[:, None] * v, 0)
    tl.store(O + bh * D + d, out)
    tl.store(Lse + bh, maximum + tl.log(denom))


@tr.jit
def decode_bwd(
    Q,
    K,
    V,
    Bias,
    O,
    Lse,
    DO,
    DQ,
    DK,
    DV,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    q = tl.load(Q + b * QS0 + h * QS1 + d).to(tl.float32)
    do = tl.load(DO + bh * D + d).to(tl.float32)
    out = tl.load(O + bh * D + d).to(tl.float32)
    lse = tl.load(Lse + bh)
    k = tl.load(K + b * KS0 + h * KS1 + n[:, None] * KS2 + d[None, :], n[:, None] < N, 0).to(tl.float32)
    bias = tl.load(Bias + b * N + n, n < N, float("-inf"))
    scores = tl.sum(k * q[None, :], 1) * SCALE + bias
    p = tl.exp(scores - lse)
    tl.store(DV + (bh * N + n[:, None]) * D + d[None, :], p[:, None] * do[None, :], n[:, None] < N)
    v = tl.load(V + b * VS0 + h * VS1 + n[:, None] * VS2 + d[None, :], n[:, None] < N, 0).to(tl.float32)
    dp = tl.sum(v * do[None, :], 1)
    delta = tl.sum(out * do, 0)
    ds = p * (dp - delta) * SCALE
    tl.store(DK + (bh * N + n[:, None]) * D + d[None, :], ds[:, None] * q[None, :], n[:, None] < N)
    dq = tl.sum(ds[:, None] * k, 0)
    tl.store(DQ + bh * D + d, dq)


@tr.jit
def _online_tile(
    q,
    K,
    V,
    Bias,
    acc,
    maxi,
    denom,
    b,
    h,
    m,
    base: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
):
    n = base + tl.arange(0, BN)
    d = tl.arange(0, D)
    k = tl.load(K + b * KS0 + h * KS1 + n[None, :] * KS2 + d[:, None], n[None, :] < N, 0)
    bias = tl.load(Bias + b * N + n, n < N, float("-inf"))
    scores = tl.dot(q, k) * (SCALE * 1.4426950408889634) + bias[None, :] * 1.4426950408889634
    newmax = tl.maximum(maxi, tl.max(scores, 1))
    safe = tl.where(newmax == float("-inf"), 0.0, newmax)
    alpha = tl.exp2(maxi - safe)
    p = tl.exp2(scores - safe[:, None])
    v = tl.load(V + b * VS0 + h * VS1 + n[:, None] * VS2 + d[None, :], n[:, None] < N, 0)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    denom = denom * alpha + tl.sum(p, 1)
    return acc, newmax, denom


@tr.jit
def fwd_stream(
    Q,
    K,
    V,
    Bias,
    O,
    Lse,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    d = tl.arange(0, D)
    q = tl.load(Q + b * QS0 + h * QS1 + m[:, None] * QS2 + d[None, :], m[:, None] < M, 0)
    acc = tl.full((BM, D), 0.0, tl.float32)
    maxi = tl.full((BM,), float("-inf"), tl.float32)
    denom = tl.full((BM,), 0.0, tl.float32)
    for i in tl.static_range(N // 128):
        acc, maxi, denom = _online_tile(
            q, K, V, Bias, acc, maxi, denom, b, h, m, i * 128, KS0, KS1, KS2, VS0, VS1, VS2, N, D, SCALE, 128
        )
    if N % 128:
        acc, maxi, denom = _online_tile(
            q, K, V, Bias, acc, maxi, denom, b, h, m, (N // 128) * 128, KS0, KS1, KS2, VS0, VS1, VS2, N, D, SCALE, BN
        )
    result = acc / denom[:, None]
    tl.store(O + ((b * M + m[:, None]) * H + h) * D + d[None, :], result, m[:, None] < M)
    tl.store(Lse + bh * M + m, (maxi + tl.log2(denom)) * 0.6931471805599453, m < M)


@tr.jit
def fwd(
    Q,
    K,
    V,
    Bias,
    O,
    Lse,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    q = tl.load(Q + b * QS0 + h * QS1 + m[:, None] * QS2 + d[None, :], m[:, None] < M, 0)
    k = tl.load(K + b * KS0 + h * KS1 + n[None, :] * KS2 + d[:, None], n[None, :] < N, 0)
    v = tl.load(V + b * VS0 + h * VS1 + n[:, None] * VS2 + d[None, :], n[:, None] < N, 0)
    bias = tl.load(Bias + b * N + n, n < N, float("-inf"))
    scores = tl.dot(q, k) * SCALE + bias[None, :]
    maximum = tl.max(scores, 1)
    p = tl.exp(scores - maximum[:, None])
    denom = tl.sum(p, 1)
    p = p / denom[:, None]
    out = tl.dot(p.to(v.dtype), v)
    tl.store(O + ((b * M + m[:, None]) * H + h) * D + d[None, :], out, m[:, None] < M)
    tl.store(Lse + bh * M + m, maximum + tl.log(denom), m < M)


@tr.jit
def dall_kernel(
    Q,
    K,
    V,
    Bias,
    O,
    Lse,
    DO,
    DQ,
    DK,
    DV,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BL: tl.constexpr,
):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    n = tl.arange(0, BL)
    d = tl.arange(0, D)
    q = tl.load(Q + b * QS0 + h * QS1 + n[:, None] * QS2 + d[None, :], n[:, None] < N, 0)
    k = tl.load(K + b * KS0 + h * KS1 + n[:, None] * KS2 + d[None, :], n[:, None] < N, 0)
    v = tl.load(V + b * VS0 + h * VS1 + n[:, None] * VS2 + d[None, :], n[:, None] < N, 0)
    do = tl.load(DO + ((b * N + n[:, None]) * H + h) * D + d[None, :], n[:, None] < N, 0)
    o = tl.load(O + ((b * N + n[:, None]) * H + h) * D + d[None, :], n[:, None] < N, 0)
    lse = tl.load(Lse + bh * N + n, n < N, 0)
    bias = tl.load(Bias + b * N + n, n < N, float("-inf"))
    p = tl.exp(tl.dot(q, tl.trans(k)) * SCALE + bias[None, :] - lse[:, None])
    p = tl.where(n[:, None] < N, p, 0.0)
    ptr = (bh * N + n[:, None]) * D + d[None, :]
    dv = tl.dot(tl.trans(p.to(do.dtype)), do)
    tl.store(DV + ptr, dv, n[:, None] < N)
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), 1)
    dp = tl.dot(do, tl.trans(v))
    ds = (p * (dp - delta[:, None]) * SCALE).to(q.dtype)
    dq = tl.dot(ds, k)
    tl.store(DQ + ptr, dq, n[:, None] < N)
    dk = tl.dot(tl.trans(ds), q)
    tl.store(DK + ptr, dk, n[:, None] < N)


@tr.jit
def dall_split_kernel(
    Q,
    K,
    V,
    Bias,
    O,
    Lse,
    DO,
    DQ,
    DK,
    DV,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    q = tl.load(Q + b * QS0 + h * QS1 + m[:, None] * QS2 + d[None, :], m[:, None] < N, 0)
    k = tl.load(K + b * KS0 + h * KS1 + n[:, None] * KS2 + d[None, :], n[:, None] < N, 0)
    v = tl.load(V + b * VS0 + h * VS1 + n[:, None] * VS2 + d[None, :], n[:, None] < N, 0)
    do = tl.load(DO + ((b * N + m[:, None]) * H + h) * D + d[None, :], m[:, None] < N, 0)
    o = tl.load(O + ((b * N + m[:, None]) * H + h) * D + d[None, :], m[:, None] < N, 0)
    lse = tl.load(Lse + bh * N + m, m < N, 0)
    bias = tl.load(Bias + b * N + n, n < N, float("-inf"))
    p = tl.exp(tl.dot(q, tl.trans(k)) * SCALE + bias[None, :] - lse[:, None])
    p = tl.where(m[:, None] < N, p, 0.0)
    ptr = (bh * N + n[:, None]) * D + d[None, :]
    dv = tl.dot(tl.trans(p.to(do.dtype)), do)
    tl.atomic_add(DV + ptr, dv, n[:, None] < N, sem="relaxed")
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), 1)
    dp = tl.dot(do, tl.trans(v))
    ds = (p * (dp - delta[:, None]) * SCALE).to(q.dtype)
    dq = tl.dot(ds, k)
    tl.store(DQ + (bh * N + m[:, None]) * D + d[None, :], dq, m[:, None] < N)
    dk = tl.dot(tl.trans(ds), q)
    tl.atomic_add(DK + ptr, dk, n[:, None] < N, sem="relaxed")


@torch.library.triton_op("puffer_wayformer::attention_forward", mutates_args=())
def _attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, queries, dim = q.shape
    keys = k.size(2)
    # Keep the following output projection in batch/token/channel order.
    out = torch.empty((batch, queries, heads, dim), dtype=q.dtype, device=q.device).transpose(1, 2)
    lse = torch.empty((batch, heads, queries), dtype=torch.float32, device=q.device)
    if queries == 1:
        wrap_triton(decode_fwd)[(batch * heads,)](
            q,
            k,
            v,
            bias,
            out,
            lse,
            *q.stride()[:2],
            *k.stride()[:3],
            *v.stride()[:3],
            heads,
            keys,
            dim,
            dim**-0.5,
            tr.next_power_of_2(keys),
            num_warps=4,
        )
    elif keys > 128:
        wrap_triton(fwd_stream)[(tr.cdiv(queries, 64), batch * heads)](
            q,
            k,
            v,
            bias,
            out,
            lse,
            *q.stride()[:3],
            *k.stride()[:3],
            *v.stride()[:3],
            heads,
            queries,
            keys,
            dim,
            dim**-0.5,
            64,
            tr.next_power_of_2(keys % 128) if keys % 128 else 128,
            num_warps=4,
        )
    else:
        block = tr.next_power_of_2(queries)
        wrap_triton(fwd)[(tr.cdiv(queries, block), batch * heads)](
            q,
            k,
            v,
            bias,
            out,
            lse,
            *q.stride()[:3],
            *k.stride()[:3],
            *v.stride()[:3],
            heads,
            queries,
            keys,
            dim,
            dim**-0.5,
            block,
            tr.next_power_of_2(keys),
            num_warps=4,
        )
    return out, lse


@torch.library.triton_op("puffer_wayformer::attention_backward", mutates_args=())
def _attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, queries, dim = q.shape
    keys = k.size(2)
    grad = grad.transpose(1, 2).contiguous().transpose(1, 2)
    dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if queries == 1 or keys <= 128:
        dk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        if queries == 1:
            wrap_triton(decode_bwd)[(batch * heads,)](
                q,
                k,
                v,
                bias,
                out,
                lse,
                grad,
                dq,
                dk,
                dv,
                *q.stride()[:2],
                *k.stride()[:3],
                *v.stride()[:3],
                heads,
                keys,
                dim,
                dim**-0.5,
                tr.next_power_of_2(keys),
                num_warps=4,
            )
        else:
            wrap_triton(dall_kernel)[(batch * heads,)](
                q,
                k,
                v,
                bias,
                out,
                lse,
                grad,
                dq,
                dk,
                dv,
                *q.stride()[:3],
                *k.stride()[:3],
                *v.stride()[:3],
                heads,
                keys,
                dim,
                dim**-0.5,
                tr.next_power_of_2(keys),
                num_warps=4,
            )
    else:
        # Query tiles contribute to shared K/V gradients. Accumulate in FP32
        # before the BF16 cast, as in other fused attention implementations.
        dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
        dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
        block = 64 if keys <= 192 else 32768 // tr.next_power_of_2(keys)
        warps = 4 if keys <= 192 else 8
        wrap_triton(dall_split_kernel)[(tr.cdiv(queries, block), batch * heads)](
            q,
            k,
            v,
            bias,
            out,
            lse,
            grad,
            dq,
            dk,
            dv,
            *q.stride()[:3],
            *k.stride()[:3],
            *v.stride()[:3],
            heads,
            keys,
            dim,
            dim**-0.5,
            block,
            tr.next_power_of_2(keys),
            num_warps=warps,
        )
        dk, dv = dk.to(k.dtype), dv.to(v.dtype)
    return dq, dk, dv


def _setup_context(ctx, inputs, output):
    out, lse = output
    ctx.save_for_backward(*inputs, out, lse)
    ctx.mark_non_differentiable(lse)


def _backward(ctx, grad, grad_lse):
    dq, dk, dv = _attention_backward(*ctx.saved_tensors, grad)
    return dq, dk, dv, None


_attention_forward.register_autograd(_backward, setup_context=_setup_context)


def attention(q, k, v, bias):
    return _attention_forward(q, k, v, bias)[0]
