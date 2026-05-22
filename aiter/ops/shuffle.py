# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import warnings
from typing import Optional, Tuple

import torch


# Tensor attribute names stamped by ``pad_weight_for_bpreshuffle`` and
# propagated by ``shuffle_weight``. These survive only as long as the
# Python tensor wrapper that carries them; ``torch.nn.Parameter(t)`` creates
# a brand-new wrapper that does not inherit ``t.__dict__`` so SGLang's
# compressed-tensors path -- which wraps every shuffled weight in
# ``nn.Parameter(..., requires_grad=False)`` -- silently strips these. The
# storage-keyed sidecar registry below is the channel that actually survives
# the wrap; the attributes are kept for ergonomics and tests that don't
# round-trip through ``Parameter``.
_BPRESHUFFLE_PADDING_ATTRS = (
    "aiter_original_k",
    "aiter_padded_k",
    "aiter_k_padding",
    "aiter_original_n",
    "aiter_padded_n",
    "aiter_n_padding",
)

# Default K/N alignment used when padding weights for the bpreshuffle GEMM.
# 256 covers all currently-shipping CK bpreshuffle BLOCK_K / NPerBlock values
# for FP8 on gfx950 regardless of whether the K-padding CK patch is active.
# With the K-padding CK patch (composable_kernel branch
# ``glm4v-fp8-padding-ck``) the K floor drops to 64 (the MFMA K-per-slot
# atomic granularity); set ``AITER_BPRESHUFFLE_PAD_ALIGNMENT=64`` to reduce
# the padded K tail from up to 168 K-rows down to <=40. The N axis still
# requires 256-alignment because the CK ``IsSupportedArgument`` check on the
# pre-shuffled layout has no analogous N-pad runtime path. If a padded shape
# is still rejected, bump to 512.
_DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT = int(
    os.environ.get("AITER_BPRESHUFFLE_PAD_ALIGNMENT", "256")
)

# Auto-padding controls whether ``shuffle_weight`` transparently pads K-dim
# and N-dim to ``_DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT`` when the input is
# unaligned. Default is on so SGLang's stock
# ``shuffle_weight(weight, (16, 16))`` call in
# compressed_tensors_w8a8_fp8.py works for GLM-4.6V at TP={2,4,8} (K
# unaligned on down_proj, N unaligned on gate_up_proj) without patching
# SGLang. Disable with ``AITER_BPRESHUFFLE_AUTO_PAD=0`` to restore the
# strict K%BK==0 / IsSupportedArgument assertion behaviour. Read inline so
# tests can flip the env via ``monkeypatch.setenv`` without reloading the
# module.
def _bpreshuffle_auto_pad_enabled() -> bool:
    return os.environ.get("AITER_BPRESHUFFLE_AUTO_PAD", "1") not in (
        "0",
        "false",
        "False",
    )


# One-shot warning sentinel; we warn only the first time auto-pad fires per
# process so logs stay clean for inference servers loading many layers.
_auto_pad_warned = False


# --- storage-keyed padding-metadata sidecar ------------------------------- #
#
# ``torch.nn.Parameter(t, requires_grad=False)`` (SGLang's
# compressed-tensors FP8 path wraps every shuffled weight this way) creates
# a fresh Python ``Tensor`` wrapper around the same underlying ``Storage``.
# Python-side attributes attached to ``t`` (``aiter_original_k`` etc.) do
# not propagate to the Parameter wrapper. The underlying ``Storage``
# (and therefore ``data_ptr()``) is preserved, so we side-channel the
# padding metadata through a process-global dict keyed on the storage's
# ``data_ptr()`` (an integer).
#
# Lifecycle: entries are inserted by ``pad_weight_for_bpreshuffle`` and by
# the auto-pad branch of ``shuffle_weight``, and never explicitly removed.
# For inference servers each shuffled weight lives for the process lifetime
# (the Parameter holds the storage), so the dict stays bounded by the
# number of FP8 linears in the loaded model -- a few hundred entries at
# most, each a ~24-byte tuple plus an int key. On stale lookups (a freed
# storage's address being reused by a new allocation) ``shape`` validation
# inside ``_lookup_bpreshuffle_padding`` rejects the entry.
_BPRESHUFFLE_PAD_REGISTRY: "dict[int, Tuple[int, int, int, int]]" = {}


def _storage_key(t: torch.Tensor) -> Optional[int]:
    """Return ``t``'s underlying storage data_ptr, or ``None`` if the tensor
    has no addressable storage (e.g. ``meta`` tensors used by torch.compile
    fake-tensor tracing)."""
    try:
        storage = t.untyped_storage()
    except (RuntimeError, AttributeError, NotImplementedError):
        return None
    try:
        ptr = storage.data_ptr()
    except (RuntimeError, AttributeError, NotImplementedError):
        return None
    return ptr if ptr != 0 else None


def _register_bpreshuffle_padding(
    t: torch.Tensor,
    original_k: int,
    padded_k: int,
    original_n: int,
    padded_n: int,
) -> None:
    """Record ``(orig_k, padded_k, orig_n, padded_n)`` for ``t``'s storage.

    No-op when no padding actually happened on either axis -- we never
    register identity entries because that would force a registry lookup
    on every aligned weight at GEMM time, and the lookup itself triggers
    a shape validation that's pointless for aligned shapes.
    """
    if padded_k == original_k and padded_n == original_n:
        return
    key = _storage_key(t)
    if key is None:
        return
    _BPRESHUFFLE_PAD_REGISTRY[key] = (original_k, padded_k, original_n, padded_n)


def _lookup_bpreshuffle_padding(
    t: torch.Tensor,
) -> Optional[Tuple[int, int, int, int]]:
    """Look up padding metadata for ``t``. Returns
    ``(orig_k, padded_k, orig_n, padded_n)`` or ``None``.

    The registry value is validated against ``t``'s current shape; a
    stale entry (e.g. after storage GC + reallocation reused the address)
    where the recorded ``padded_*`` no longer matches ``t.shape`` is
    treated as a miss. This is the safety net that lets us key on
    ``data_ptr()`` without a per-tensor weakref.
    """
    key = _storage_key(t)
    if key is None:
        return None
    entry = _BPRESHUFFLE_PAD_REGISTRY.get(key)
    if entry is None:
        return None
    _, padded_k, _, padded_n = entry
    if t.shape[-1] != padded_k or t.shape[-2] != padded_n:
        return None
    return entry


def pad_weight_for_bpreshuffle(
    x: torch.Tensor,
    alignment: int = _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT,
    layout=(16, 16),
    pad_n: bool = False,
) -> torch.Tensor:
    """Right-pad ``K`` (last dim) -- and optionally ``N`` (second-to-last) --
    to a multiple of ``alignment`` with zeros.

    Returns a contiguous tensor tagged with ``aiter_original_k``,
    ``aiter_padded_k`` and ``aiter_k_padding`` (and, when ``pad_n=True``,
    the matching ``aiter_*_n`` attrs) and *also* registered in the
    storage-keyed sidecar (so the metadata survives a subsequent
    ``torch.nn.Parameter`` wrap).

    The padded tail is exactly zero. For per-token / per-channel FP8 quant
    this keeps the GEMM result mathematically identical: ``x_scale`` is per
    token (no K dim) and ``w_scale`` is per output channel (so K-padding
    leaves it untouched; N-padding requires the matching ``w_scale``
    N-extension which ``gemm_a8w8_bpreshuffle`` handles automatically using
    the sidecar metadata).

    ``alignment=256`` covers all currently-shipping CK bpreshuffle
    ``BLOCK_K`` / ``NPerBlock`` values for FP8 on gfx950. If a padded shape
    is still rejected by ``IsSupportedArgument``, bump via
    ``AITER_BPRESHUFFLE_PAD_ALIGNMENT=512`` and rerun the tuner.

    Memory peak during this helper is ``2 * sizeof(weight)`` because both
    ``x`` and ``out`` are live until return. Callers that load weights
    eagerly should ``del`` the original tensor and call
    ``torch.cuda.empty_cache()`` once the returned tensor has been wrapped
    in a ``Parameter``.
    """
    if alignment <= 0:
        raise ValueError(
            f"pad_weight_for_bpreshuffle: alignment must be positive, got "
            f"{alignment}"
        )

    original_k = x.shape[-1]
    padded_k = ((original_k + alignment - 1) // alignment) * alignment
    original_n = x.shape[-2] if x.ndim >= 2 else 1
    if pad_n and x.ndim >= 2:
        padded_n = ((original_n + alignment - 1) // alignment) * alignment
    else:
        padded_n = original_n

    if padded_k == original_k and padded_n == original_n:
        out = x.contiguous()
    elif padded_n == original_n:
        # K-only pad: avoid a full zero pass over ``original_k`` elements
        # by allocating with ``empty`` and zeroing only the K tail.
        out = torch.empty(
            *x.shape[:-1], padded_k, dtype=x.dtype, device=x.device
        )
        out[..., :original_k].copy_(x)
        out[..., original_k:].zero_()
    else:
        # N (and possibly K) pad: allocate the full padded box with zeros
        # then copy the original block. A separate ``zero_()`` of the
        # bottom/right L-shaped tail would need two strided writes; one
        # ``zeros`` allocation is simpler and not measurably slower at
        # the sizes we care about (<=64MB per weight).
        out = torch.zeros(
            *x.shape[:-2], padded_n, padded_k, dtype=x.dtype, device=x.device
        )
        out[..., :original_n, :original_k].copy_(x)

    out.aiter_original_k = original_k
    out.aiter_padded_k = padded_k
    out.aiter_k_padding = padded_k - original_k
    out.aiter_original_n = original_n
    out.aiter_padded_n = padded_n
    out.aiter_n_padding = padded_n - original_n
    _register_bpreshuffle_padding(out, original_k, padded_k, original_n, padded_n)
    return out


def _propagate_bpreshuffle_padding_attrs(
    src: torch.Tensor, dst: torch.Tensor
) -> None:
    """Copy padding attrs from ``src`` to ``dst`` and re-register the
    sidecar at ``dst``'s storage.

    Re-registration is necessary because operations such as ``.contiguous()``
    after a non-trivial permutation -- the core of the shuffle path --
    allocate a new storage, so ``dst`` has a different ``data_ptr()`` from
    ``src`` even though it carries the same logical metadata.
    """
    for attr in _BPRESHUFFLE_PADDING_ATTRS:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))

    # Re-register dst in the sidecar so the metadata is reachable after
    # ``torch.nn.Parameter`` wrapping (Parameter shares storage with its
    # input, so ``dst.untyped_storage().data_ptr()`` matches the wrapped
    # Parameter's ``data_ptr()``).
    src_entry = _lookup_bpreshuffle_padding(src)
    if src_entry is None and hasattr(src, "aiter_original_k"):
        # Fall back to attrs (the helper above stamps both attrs and the
        # sidecar, but external callers might stamp only attrs).
        src_entry = (
            getattr(src, "aiter_original_k", src.shape[-1]),
            getattr(src, "aiter_padded_k", src.shape[-1]),
            getattr(src, "aiter_original_n", src.shape[-2]),
            getattr(src, "aiter_padded_n", src.shape[-2]),
        )
    if src_entry is not None:
        original_k, padded_k, original_n, padded_n = src_entry
        _register_bpreshuffle_padding(
            dst, original_k, padded_k, original_n, padded_n
        )


def shuffle_weight(
    x: torch.Tensor,
    layout=(16, 16),
    use_int4=False,
    is_guinterleave=False,
    gate_up: bool = False,
) -> torch.Tensor:
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    if is_guinterleave:
        experts_cnt, N, K_pk = x.shape
        if gate_up:
            N = N // 2
        NLane, KPack = layout
        KLane = 64 // NLane
        N0 = N // NLane
        K0 = K_pk // (KLane * KPack)
        if gate_up:
            x_ = x.view(experts_cnt, 2, N0, NLane, K0, KLane, KPack)
            x_ = x_.permute(0, 2, 1, 4, 5, 3, 6).contiguous()
        else:
            x_ = x.view(experts_cnt, N0, NLane, K0, KLane, KPack)
            x_ = x_.permute(0, 1, 3, 4, 2, 5).contiguous()
        x_ = x_.view(*x.shape).contiguous().view(x_type)
        x_.is_shuffled = True
        _propagate_bpreshuffle_padding_attrs(x, x_)
        return x_

    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN

    # Transparent K/N-padding for unaligned inputs. Two GLM-4.6V FP8 cases:
    #
    #   * K unaligned: down_proj at TP={2,4,8} has per-shard K =
    #     intermediate_size / TP = {5472, 2736, 1368} which trip the
    #     ``K % BK == 0`` assertion below (BK=32, none of those K values
    #     are multiples of 32... wait, 5472 IS a multiple of 32 so the
    #     pre-pad assertion passes for TP=2 -- but the downstream CK
    #     ``IsSupportedArgument`` still rejects intra-K0-slot tails. We
    #     pad K to 256 unconditionally so all three TP cases satisfy both
    #     the AIter shuffle assertion AND the CK dispatch).
    #   * N unaligned: gate_up_proj at TP={4,8} has per-shard
    #     N = 2 * intermediate_size / TP = {5472, 2736}; both pass the
    #     ``N % BN == 0`` shuffle assertion (BN=16) but no CK bpreshuffle
    #     instance has an ``NPerBlock`` that divides them, so dispatch
    #     fails with ``"This GEMM is not supported!"``. Padding N to a
    #     multiple of 256 makes the CK instance list match.
    #
    # Activation matching (XQ right-pad on K) and output slicing (Y left-
    # truncate on N) happen inside ``gemm_a8w8_bpreshuffle``, which reads
    # the metadata through the storage-keyed sidecar registered by
    # ``_propagate_bpreshuffle_padding_attrs`` below.
    auto_pad = _bpreshuffle_auto_pad_enabled()
    k_unaligned = x.shape[-1] % _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT != 0
    # Only auto-pad N for "large" weights -- small N (<= alignment) goes
    # through CK's small-tile instances that don't require alignment to
    # the NPerBlock of 64+. The GLM-4.6V N values we need to fix are all
    # in the thousands, well above this threshold. Without this guard
    # tiny test fixtures (N=4, N=16, N=32) would silently bloat to 256
    # rows and break tests that introspect ``aiter_padded_n``.
    n_unaligned = (
        x.shape[-2] > _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT
        and x.shape[-2] % _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT != 0
    )
    if auto_pad and (k_unaligned or n_unaligned):
        global _auto_pad_warned
        pad_align = _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT
        new_k = (
            ((x.shape[-1] + pad_align - 1) // pad_align) * pad_align
            if k_unaligned
            else x.shape[-1]
        )
        new_n = (
            ((x.shape[-2] + pad_align - 1) // pad_align) * pad_align
            if n_unaligned
            else x.shape[-2]
        )
        if not _auto_pad_warned:
            warnings.warn(
                "[aiter] shuffle_weight auto-padded "
                f"K={x.shape[-1]}->{new_k}, N={x.shape[-2]}->{new_n} "
                f"(next multiple of {pad_align}). Padded tails are zero; "
                "activations are right-padded and outputs are left-sliced "
                "inside gemm_a8w8_bpreshuffle. Disable with "
                "AITER_BPRESHUFFLE_AUTO_PAD=0 (strict assertion mode) or "
                "tighten the K alignment with "
                "AITER_BPRESHUFFLE_PAD_ALIGNMENT=64 when the K-padding CK "
                "patch is active. Suppressing further auto-pad warnings.",
                stacklevel=2,
            )
            _auto_pad_warned = True
        # ``pad_n`` is gated by the magnitude threshold so small-N weights
        # whose K trips ``k_unaligned`` don't get their N silently bloated.
        x = pad_weight_for_bpreshuffle(x, layout=layout, pad_n=n_unaligned)

    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    x_ = x_.view(x_type)
    x_.is_shuffled = True
    _propagate_bpreshuffle_padding_attrs(x, x_)
    return x_


def is_bpreshuffle_kernel_tuned(
    n: int,
    k: int,
    dtype: torch.dtype,
    libtype: Optional[str] = None,
    m_candidates: Optional[list] = None,
) -> bool:
    """Return True if the bpreshuffle tuned CSV has at least one entry that
    matches ``(N=n, K=k, q_dtype_w=dtype)`` for the active GFX/CU.

    Used by SGLang at weight-load time to decide whether to pad-and-shuffle
    a weight (fast path) or fall back to the unshuffled
    ``gemm_a8w8_CK`` kernel for that layer. Probing a representative set of
    M values mirrors the M sweep used during tuning so layers whose tuned M
    range is sparse still resolve to True.

    ``libtype`` filters by the ``libtype`` column (e.g. ``"ck"``,
    ``"cktile"``, ``"flydsl"``). Default of ``None`` returns True for any
    libtype.
    """
    # Imported lazily to avoid an import cycle: ``gemm_op_a8w8`` imports from
    # ``..jit.core`` which transitively imports many AIter ops.
    from .gemm_op_a8w8 import get_GEMM_config_with_quant_type
    from ..jit.core import AITER_CONFIGS

    if m_candidates is None:
        # Mirrors the SGLang graph-capture batch-size sweep documented in the
        # GLM-4.6V FP8 padding plan.
        m_candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    for m in m_candidates:
        config = get_GEMM_config_with_quant_type(
            m,
            n,
            k,
            dtype,
            AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
        )
        if config is None:
            continue
        if libtype is None or config.get("libtype") == libtype:
            return True
    return False


def shuffle_weight_a16w4(src: torch.Tensor, NLane: int, gate_up: bool) -> torch.Tensor:
    """Backward-compatible wrapper around `shuffle_weight(..., is_guinterleave=True)`."""
    return shuffle_weight(
        src, layout=(NLane, 16), is_guinterleave=True, gate_up=gate_up
    )


def shuffle_weight_NK(
    x: torch.Tensor, inst_N: int, inst_K: int, use_int4=False
) -> torch.Tensor:
    kPerLane = inst_K // (64 // inst_N)
    if use_int4:
        kPerLane *= 2
    assert (
        x.shape[-2] % inst_N == 0
    ), f"{x.shape[-2]} % {inst_N} == {x.shape[-2] % inst_N }"
    assert (
        x.shape[-1] % inst_K == 0
    ), f"{x.shape[-1]} % {inst_K} == {x.shape[-1] % inst_K }"

    x_ = x
    x_ = x_.view(
        -1, x.shape[-2] // inst_N, inst_N, x.shape[-1] // inst_K, 64 // inst_N, kPerLane
    )
    x_ = x_.permute(0, 1, 3, 4, 2, 5).contiguous()
    return x_.view(*x.shape)


def shuffle_scale(
    src: torch.Tensor,
    experts_cnt: int = None,
    is_guinterleave: bool = False,
    gate_up: bool = False,
) -> torch.Tensor:
    if src is None:
        return src
    if src.dtype == torch.float32:
        return src
    assert src.ndim == 2, "scale must be a 2D tensor"

    if not is_guinterleave:
        m, n = src.shape
        scale_padded = torch.empty(
            (m + 255) // 256 * 256,
            (n + 7) // 8 * 8,
            dtype=src.dtype,
            device=src.device,
        )

        scale_padded[:m, :n] = src
        scale = scale_padded
        sm, sn = scale.shape
        scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
        scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
        return scale.view(sm, sn)

    if experts_cnt is None:
        raise ValueError("experts_cnt is required when is_guinterleave=True")

    n_experts, k_ = src.shape
    n_ = n_experts // experts_cnt
    # MXFP4 constants
    K_Pack = 2
    N_Pack = 2
    N_Lane = 16
    K_Lane = 64 // N_Lane  # 4

    # Basic dimensions
    K1 = k_ // K_Pack // K_Lane  # k_ // 8
    N1 = n_ // N_Lane // N_Pack  # n_ // 32
    real_k = 32 * k_ * K_Pack * K_Lane  # 1x32 quant
    assert real_k >= 256, f"K {real_k} must be larger than Tile_K(256)"
    # print("src shape", src.shape)
    # Reshape based on moe_kind
    if gate_up:
        # Reshape to: [E, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane]
        shfl_scale = src.view(experts_cnt, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane)
        # Permute to: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
        shfl_scale = shfl_scale.permute(0, 2, 4, 6, 3, 5, 1).contiguous()
    else:
        # Reshape to: [E, K1, K_Pack, K_Lane, N1, N_Pack, N_Lane]
        shfl_scale = src.view(experts_cnt, N1, N_Pack, N_Lane, K1, K_Pack, K_Lane)
        # Permute to: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
        shfl_scale = shfl_scale.permute(0, 1, 4, 6, 3, 5, 2).contiguous()
    # print("shf_scale shape:", shfl_scale.shape)
    return shfl_scale.view(*src.shape).contiguous()


def shuffle_scale_a16w4(
    src: torch.Tensor, experts_cnt: int, gate_up: bool
) -> torch.Tensor:
    """Backward-compatible wrapper around `shuffle_scale(..., is_guinterleave=True)`."""
    return shuffle_scale(
        src, experts_cnt=experts_cnt, is_guinterleave=True, gate_up=gate_up
    )


def pack_int8_to_packed_int4(x_shuf_i8: torch.Tensor) -> torch.Tensor:
    """Pack a preshuffled int8 tensor (values in [-8, 7]) into packed int4 bytes.

    Each contiguous 8-value block [v0..v7] -> 4 bytes:
      b0=(v4<<4)|v0, b1=(v5<<4)|v1, b2=(v6<<4)|v2, b3=(v7<<4)|v3.

    This matches the 7-op in-kernel unpack sequence used by FlyDSL int4_bf16.
    """
    flat = x_shuf_i8.contiguous().view(-1).to(torch.int16)
    assert flat.numel() % 8 == 0
    u = (flat & 0xF).to(torch.uint8).view(-1, 8)
    out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
    out[:, 0] = u[:, 0] | (u[:, 4] << 4)
    out[:, 1] = u[:, 1] | (u[:, 5] << 4)
    out[:, 2] = u[:, 2] | (u[:, 6] << 4)
    out[:, 3] = u[:, 3] | (u[:, 7] << 4)
    return out.view(-1).to(torch.int8)


def shuffle_scale_for_int4(scale: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Prepare groupwise scale tensor for W4A16 int4 kernel.

    Input: scale tensor of shape ``[E, num_groups, N]``.

    For **f32** scales the kernel uses ``(E, G, N)`` layout directly.

    For **bf16** scales the kernel uses ``(E, G//2, N, 2)`` layout -- two
    adjacent groups for the same N position are packed into one dword.

    Only group_size=32 is supported due to int4 preshuffle layout constraints.
    """
    if group_size != 32:
        raise ValueError(
            f"shuffle_scale_for_int4 only supports group_size=32, got {group_size}. "
            f"This is due to int4 preshuffle layout constraints."
        )

    if scale.dtype == torch.bfloat16:
        E, G, N = scale.shape
        return scale.view(E, G // 2, 2, N).permute(0, 1, 3, 2).contiguous()

    return scale.contiguous()
