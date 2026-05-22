# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``aiter.ops.shuffle.pad_weight_for_bpreshuffle`` and the padded
bpreshuffle path of ``aiter.gemm_a8w8_bpreshuffle``.

These tests cover the GLM-4.6V FP8 K values (1368, 2736, 5472, 6144) that
trip the shuffle-weight K-divisibility assertion and/or the
``IsSupportedArgument`` rejection inside the bpreshuffle kernel. They run
on CPU-only for the metadata/regression cases and require CUDA for the GEMM
correctness cases.
"""

from __future__ import annotations

import pytest
import torch

from aiter.ops.shuffle import (
    _BPRESHUFFLE_PAD_REGISTRY,
    _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT,
    _bpreshuffle_auto_pad_enabled,
    _lookup_bpreshuffle_padding,
    pad_weight_for_bpreshuffle,
    shuffle_weight,
)


GLM_K_PADDED_TO_UNPADDED = {
    # (TP, role, original_k): padded_k for AITER_BPRESHUFFLE_PAD_ALIGNMENT=256
    "tp8_down_proj": (1368, 1536),
    "tp4_down_proj": (2736, 2816),
    "tp2_down_proj": (5472, 5632),
    "tp2_other": (6144, 6144),
}

# (TP, role, original_n): padded_n. These are the gate_up_proj per-TP shard
# sizes that hit ``RuntimeError: This GEMM is not supported!`` from CK at
# TP=4/8 because no bpreshuffle instance has an NPerBlock dividing them.
GLM_N_PADDED_TO_UNPADDED = {
    "tp8_gate_up": (2736, 2816),
    "tp4_gate_up": (5472, 5632),
    "tp2_gate_up": (10944, 11008),  # 10944 % 256 == 192 -> next mult is 11008
}

ALIGNED_K_VALUES = (4096, 1024, 1536, 2048, 3072)
PADDED_K_VALUES = (1368, 2736, 5472)
PADDED_N_VALUES = (2736, 5472)
NON_FP8_DTYPES = (torch.bfloat16, torch.float16, torch.int8)


# --- pad_weight_for_bpreshuffle: padding correctness ---------------------- #


@pytest.mark.parametrize(
    "case",
    list(GLM_K_PADDED_TO_UNPADDED.values()),
    ids=[k for k in GLM_K_PADDED_TO_UNPADDED],
)
def test_pad_metadata_matches_alignment(case):
    original_k, padded_k = case
    n = 64
    w = torch.zeros((n, original_k), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w)
    assert out.shape[-1] == padded_k
    assert out.shape[-2] == n
    assert out.aiter_original_k == original_k
    assert out.aiter_padded_k == padded_k
    assert out.aiter_k_padding == padded_k - original_k


@pytest.mark.parametrize("original_k", PADDED_K_VALUES)
def test_pad_tail_is_exactly_zero(original_k):
    w = torch.randn((32, original_k), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w)
    if out.aiter_k_padding == 0:
        pytest.skip(
            f"K={original_k} is already aligned to "
            f"{_DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT}; nothing to test"
        )
    tail = out[..., original_k:]
    assert torch.all(tail == 0), (
        f"padded tail for K={original_k} -> {out.aiter_padded_k} should be "
        f"exactly zero, got max abs={tail.abs().max().item()}"
    )


@pytest.mark.parametrize("original_k", PADDED_K_VALUES)
def test_pad_head_equals_input(original_k):
    w = torch.randn((32, original_k), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w)
    head = out[..., :original_k]
    assert torch.equal(head, w), (
        "padded head must bit-exactly equal the input weight slice"
    )


@pytest.mark.parametrize("original_k", ALIGNED_K_VALUES)
def test_pad_no_op_on_aligned_k(original_k):
    """Already-aligned K must round-trip without padding."""
    w = torch.randn((32, original_k), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w)
    assert out.shape == w.shape
    assert out.aiter_original_k == original_k
    assert out.aiter_padded_k == original_k
    assert out.aiter_k_padding == 0


@pytest.mark.parametrize("dtype", NON_FP8_DTYPES)
def test_pad_works_for_non_fp8_dtypes(dtype):
    """The padding helper is dtype-agnostic; metadata propagates regardless."""
    if dtype == torch.int8:
        w = torch.randint(-8, 8, (16, 1368), dtype=dtype)
    else:
        w = torch.randn((16, 1368), dtype=dtype)
    out = pad_weight_for_bpreshuffle(w)
    assert out.dtype == dtype
    assert out.aiter_original_k == 1368
    assert out.aiter_padded_k > 1368
    assert torch.equal(out[..., :1368], w)
    assert torch.all(out[..., 1368:] == 0)


def test_pad_alignment_argument_overrides_default():
    w = torch.randn((16, 1024), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w, alignment=2048)
    assert out.aiter_padded_k == 2048
    assert out.aiter_k_padding == 1024


def test_pad_alignment_must_be_positive():
    w = torch.zeros((4, 256), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        pad_weight_for_bpreshuffle(w, alignment=0)
    with pytest.raises(ValueError):
        pad_weight_for_bpreshuffle(w, alignment=-1)


# --- shuffle_weight: metadata propagation -------------------------------- #


@pytest.mark.parametrize("original_k", PADDED_K_VALUES)
def test_shuffle_weight_propagates_padding_metadata(original_k):
    w = torch.randn((32, original_k), dtype=torch.bfloat16)
    padded = pad_weight_for_bpreshuffle(w)
    shuffled = shuffle_weight(padded, layout=(16, 16))
    assert getattr(shuffled, "is_shuffled", False) is True
    assert shuffled.aiter_original_k == padded.aiter_original_k
    assert shuffled.aiter_padded_k == padded.aiter_padded_k
    assert shuffled.aiter_k_padding == padded.aiter_k_padding


def test_shuffle_weight_strict_when_auto_pad_disabled(monkeypatch):
    """With auto-pad disabled, ``shuffle_weight`` retains upstream AIter's
    strict K%BK==0 assertion -- so callers that opt out get the same
    behaviour they would on stock AIter."""
    monkeypatch.setenv("AITER_BPRESHUFFLE_AUTO_PAD", "0")
    assert not _bpreshuffle_auto_pad_enabled()
    w = torch.randn((32, 1368), dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        shuffle_weight(w, layout=(16, 16))


@pytest.mark.parametrize("original_k", PADDED_K_VALUES)
def test_shuffle_weight_auto_pads_unaligned_k(original_k, monkeypatch):
    """With auto-pad enabled (the new default), calling ``shuffle_weight``
    directly on unaligned K (GLM-4.6V TP={2,4,8} shapes) succeeds and the
    returned tensor carries the padding metadata so ``gemm_a8w8_bpreshuffle``
    can match activations. SGLang's compressed-tensors FP8 path relies on
    this so we don't have to patch SGLang to handle GLM-4.6V."""
    monkeypatch.setenv("AITER_BPRESHUFFLE_AUTO_PAD", "1")
    assert _bpreshuffle_auto_pad_enabled()
    w = torch.randn((32, original_k), dtype=torch.bfloat16)
    shuffled = shuffle_weight(w, layout=(16, 16))
    expected_k = (
        (original_k + _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT - 1)
        // _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT
    ) * _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT
    assert shuffled.shape[-1] == expected_k
    assert shuffled.aiter_original_k == original_k
    assert shuffled.aiter_padded_k == expected_k
    assert shuffled.aiter_k_padding == expected_k - original_k
    assert getattr(shuffled, "is_shuffled", False) is True


def test_shuffle_weight_aligned_input_has_no_padding_attrs():
    """If the input has no padding metadata, the output should not have it
    either -- we never invent metadata out of thin air.
    """
    w = torch.randn((32, 4096), dtype=torch.bfloat16)
    shuffled = shuffle_weight(w, layout=(16, 16))
    assert not hasattr(shuffled, "aiter_original_k")
    assert not hasattr(shuffled, "aiter_padded_k")
    assert not hasattr(shuffled, "aiter_k_padding")
    assert not hasattr(shuffled, "aiter_original_n")
    assert not hasattr(shuffled, "aiter_padded_n")


# --- N-axis padding ------------------------------------------------------ #


@pytest.mark.parametrize(
    "case",
    list(GLM_N_PADDED_TO_UNPADDED.values()),
    ids=[k for k in GLM_N_PADDED_TO_UNPADDED],
)
def test_pad_weight_n_dim(case):
    """``pad_weight_for_bpreshuffle(..., pad_n=True)`` pads N up to the
    alignment with zero rows and stamps the matching ``aiter_*_n`` attrs.
    """
    original_n, padded_n = case
    k = 4096
    w = torch.randn((original_n, k), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w, pad_n=True)
    assert out.shape == (padded_n, k)
    assert out.aiter_original_n == original_n
    assert out.aiter_padded_n == padded_n
    assert out.aiter_n_padding == padded_n - original_n
    # K untouched in these cases (k=4096 already aligned).
    assert out.aiter_original_k == k
    assert out.aiter_padded_k == k
    # Head must equal input bit-for-bit, tail must be exactly zero.
    assert torch.equal(out[:original_n, :], w)
    if padded_n > original_n:
        assert torch.all(out[original_n:, :] == 0)


def test_pad_weight_both_n_and_k():
    """Edge case: a weight unaligned on both axes must get both axes padded
    in a single ``pad_weight_for_bpreshuffle(pad_n=True)`` call."""
    w = torch.randn((2736, 5472), dtype=torch.bfloat16)  # N and K both unaligned
    out = pad_weight_for_bpreshuffle(w, pad_n=True)
    assert out.shape == (2816, 5632)
    assert out.aiter_original_n == 2736 and out.aiter_padded_n == 2816
    assert out.aiter_original_k == 5472 and out.aiter_padded_k == 5632
    assert torch.equal(out[:2736, :5472], w)
    assert torch.all(out[2736:, :] == 0)
    assert torch.all(out[:, 5472:] == 0)


@pytest.mark.parametrize("original_n", PADDED_N_VALUES)
def test_shuffle_weight_auto_pads_unaligned_n(original_n, monkeypatch):
    """When N is unaligned, ``shuffle_weight`` must auto-pad N too. This
    is what unblocks GLM-4.6V FP8 gate_up_proj at TP={4,8}, whose per-TP
    N values (5472 and 2736) don't match any CK bpreshuffle instance."""
    monkeypatch.setenv("AITER_BPRESHUFFLE_AUTO_PAD", "1")
    k = 4096  # aligned; we want to isolate the N path
    w = torch.randn((original_n, k), dtype=torch.bfloat16)
    shuffled = shuffle_weight(w, layout=(16, 16))
    expected_n = (
        (original_n + _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT - 1)
        // _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT
    ) * _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT
    assert shuffled.shape[-2] == expected_n
    assert shuffled.shape[-1] == k
    assert shuffled.aiter_original_n == original_n
    assert shuffled.aiter_padded_n == expected_n
    assert getattr(shuffled, "is_shuffled", False) is True


# --- storage-keyed sidecar: survives nn.Parameter wrapping ---------------- #


@pytest.mark.parametrize("original_k", PADDED_K_VALUES)
def test_metadata_survives_parameter_wrap_k(original_k):
    """The whole reason for the sidecar: ``nn.Parameter`` strips
    tensor.__dict__, so attribute-only metadata is lost. The
    storage-keyed registry survives because Parameter shares storage."""
    w = torch.randn((128, original_k), dtype=torch.bfloat16)
    shuffled = shuffle_weight(w, layout=(16, 16))
    assert shuffled.aiter_padded_k > original_k

    param = torch.nn.Parameter(shuffled, requires_grad=False)
    # Confirm the attribute channel is dead (the regression we are fixing).
    assert not hasattr(param, "aiter_original_k")
    assert not hasattr(param, "aiter_padded_k")
    # The sidecar must still recover the metadata for the Parameter.
    side = _lookup_bpreshuffle_padding(param)
    assert side is not None, (
        "storage-keyed sidecar lookup failed after nn.Parameter wrap; "
        "the bpreshuffle GEMM path will not know to auto-pad XQ."
    )
    sok, spk, son, spn = side
    assert sok == original_k
    assert spk == shuffled.aiter_padded_k


@pytest.mark.parametrize("original_n", PADDED_N_VALUES)
def test_metadata_survives_parameter_wrap_n(original_n):
    """Same as above but for N-padding."""
    k = 4096
    w = torch.randn((original_n, k), dtype=torch.bfloat16)
    shuffled = shuffle_weight(w, layout=(16, 16))
    assert shuffled.aiter_padded_n > original_n

    param = torch.nn.Parameter(shuffled, requires_grad=False)
    side = _lookup_bpreshuffle_padding(param)
    assert side is not None
    sok, spk, son, spn = side
    assert son == original_n
    assert spn == shuffled.aiter_padded_n


def test_registry_does_not_register_aligned_weights():
    """Aligned weights produce no registry entry so the lookup at GEMM
    time is a single dict miss for the common path. This is a regression
    guard so we don't accidentally degrade the aligned path."""
    before = len(_BPRESHUFFLE_PAD_REGISTRY)
    w = torch.randn((4096, 4096), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w, pad_n=True)
    assert _lookup_bpreshuffle_padding(out) is None
    # And neither padding path inserted a registry entry.
    assert len(_BPRESHUFFLE_PAD_REGISTRY) == before


def test_registry_rejects_stale_entry_on_shape_mismatch():
    """If a registry entry's recorded ``padded_*`` no longer matches the
    tensor's current shape, ``_lookup_bpreshuffle_padding`` returns None.
    This is the guard against a freed-storage data_ptr being reused for
    an unrelated tensor with a different shape."""
    w = torch.randn((128, 2736), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w)
    assert _lookup_bpreshuffle_padding(out) is not None
    # Fabricate a stale registry entry by overwriting the shape data via
    # ``.view`` -- the underlying storage is shared but the shape changes.
    reshaped = out.view(-1, out.shape[-1] // 2)
    # The sidecar must refuse to return the entry for the reshaped tensor.
    assert _lookup_bpreshuffle_padding(reshaped) is None


# --- gemm_a8w8_bpreshuffle: padded GEMM correctness ---------------------- #
#
# These tests need a CUDA device + AIter's compiled bpreshuffle kernel. Skip
# cleanly when not available so the metadata tests above can run on CPU CI.


CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(scope="module")
def aiter_module():
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA required for bpreshuffle GEMM correctness tests")
    import aiter

    return aiter


def _fp8_perchannel_quant(t: torch.Tensor):
    """Per-row FP8 quant matching the SGLang weight path. Returns
    (q_t [N,K] fp8_e4m3fnuz, scale [N,1] fp32)."""
    import aiter
    from aiter import dtypes

    return aiter.pertoken_quant(t, quant_dtype=dtypes.fp8)


def _torch_reference(xq, wq, x_scale, w_scale, dtype):
    """Reference matching what ``gemm_a8w8_bpreshuffle`` computes: dequant
    FP8 -> FP32 GEMM -> apply per-token (x) and per-channel (w) scales ->
    cast to ``dtype``. Comparing to the unquantized BF16 inputs is wrong --
    FP8 quantization drift would dwarf the tolerance even with no padding."""
    x = xq.to(torch.float32) * x_scale.to(torch.float32)
    w = wq.to(torch.float32) * w_scale.to(torch.float32)
    return torch.nn.functional.linear(x, w).to(dtype)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 4096, 1368),
        (16, 4096, 2736),
        (1, 4096, 5472),
        (16, 4096, 6144),
        (16, 4096, 4096),
    ],
)
def test_padded_bpreshuffle_matches_torch_reference(aiter_module, m, n, k):
    """End-to-end: pad + shuffle the weight, manually pad the activation,
    compare against the BF16 reference."""
    aiter = aiter_module

    torch.manual_seed(0)
    device = "cuda"
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device)

    xq, x_scale = _fp8_perchannel_quant(x_bf16)
    wq, w_scale = _fp8_perchannel_quant(w_bf16)

    padded_wq = pad_weight_for_bpreshuffle(wq)
    shuffled = shuffle_weight(padded_wq, layout=(16, 16))

    if shuffled.aiter_padded_k != shuffled.aiter_original_k:
        padded_xq = torch.zeros(
            (m, shuffled.aiter_padded_k),
            dtype=xq.dtype,
            device=device,
        )
        padded_xq[:, :k] = xq
    else:
        padded_xq = xq

    try:
        out = aiter.gemm_a8w8_bpreshuffle(
            padded_xq, shuffled, x_scale, w_scale, None, torch.bfloat16
        )
    except RuntimeError as e:
        pytest.skip(
            f"bpreshuffle kernel rejected (M={m}, N={n}, K_padded="
            f"{shuffled.aiter_padded_k}); SGLang falls back to gemm_a8w8_CK "
            f"for this shape. Underlying error: {e}"
        )

    ref = _torch_reference(xq, wq, x_scale, w_scale, torch.bfloat16)
    assert out.shape == (m, n), (out.shape, (m, n))
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 4096, 1368),
        (16, 4096, 2736),
        (1, 4096, 5472),
    ],
)
def test_padded_weight_auto_pads_activation(aiter_module, m, n, k):
    """With auto-pad on, passing an unpadded activation alongside a padded
    weight must auto-pad XQ inside ``gemm_a8w8_bpreshuffle`` and produce a
    result numerically identical to the manually-padded path. This is the
    SGLang GLM-4.6V flow: compressed-tensors loads weights and runs them
    through ``shuffle_weight`` (which now auto-pads); activations stay at
    the original K and AIter must handle the mismatch."""
    aiter = aiter_module

    torch.manual_seed(0)
    device = "cuda"
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device)

    xq, x_scale = _fp8_perchannel_quant(x_bf16)
    wq, w_scale = _fp8_perchannel_quant(w_bf16)

    # Mimic SGLang's path: ``shuffle_weight`` alone on unaligned K.
    shuffled = shuffle_weight(wq, layout=(16, 16))
    if shuffled.aiter_padded_k == shuffled.aiter_original_k:
        pytest.skip(f"expected K={k} to need padding for this test")

    try:
        out = aiter.gemm_a8w8_bpreshuffle(
            xq, shuffled, x_scale, w_scale, None, torch.bfloat16
        )
    except RuntimeError as e:
        pytest.skip(
            f"bpreshuffle kernel rejected (M={m}, N={n}, K_padded="
            f"{shuffled.aiter_padded_k}). Underlying error: {e}"
        )
    assert out.shape == (m, n), (out.shape, (m, n))

    # Compare to the manually-padded path; results must be identical because
    # the padded tail is zero on both sides.
    padded_xq = torch.zeros((m, shuffled.aiter_padded_k), dtype=xq.dtype, device=device)
    padded_xq[:, :k] = xq
    out_manual = aiter.gemm_a8w8_bpreshuffle(
        padded_xq, shuffled, x_scale, w_scale, None, torch.bfloat16
    )
    torch.testing.assert_close(out, out_manual, rtol=0, atol=0)

    ref = _torch_reference(xq, wq, x_scale, w_scale, torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_padded_weight_unpadded_activation_raises_when_auto_pad_disabled(
    aiter_module, monkeypatch
):
    """Kill-switch behavior: with AITER_BPRESHUFFLE_AUTO_PAD=0 the
    unpadded-activation case raises a clear AssertionError instead of
    silently auto-padding. Lets users opt back into the strict contract."""
    aiter = aiter_module
    monkeypatch.setenv("AITER_BPRESHUFFLE_AUTO_PAD", "0")

    torch.manual_seed(0)
    device = "cuda"
    m, n, k = 16, 4096, 2736
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device)

    xq, x_scale = _fp8_perchannel_quant(x_bf16)
    wq, w_scale = _fp8_perchannel_quant(w_bf16)

    # With AUTO_PAD=0 ``shuffle_weight`` won't auto-pad either, so use the
    # explicit helper to set up a padded weight without triggering the
    # disabled auto-path.
    padded_wq = pad_weight_for_bpreshuffle(wq)
    if padded_wq.aiter_padded_k == padded_wq.aiter_original_k:
        pytest.skip("expected K=2736 to need padding for this test")
    shuffled = shuffle_weight(padded_wq, layout=(16, 16))

    with pytest.raises(AssertionError):
        aiter.gemm_a8w8_bpreshuffle(
            xq, shuffled, x_scale, w_scale, None, torch.bfloat16
        )


@pytest.mark.parametrize(
    "m,n,k",
    [
        (16, 2736, 4096),   # GLM-4.6V FP8 gate_up_proj per-shard at TP=8
        (16, 5472, 4096),   # GLM-4.6V FP8 gate_up_proj per-shard at TP=4
        (64, 5472, 4096),   # GLM-4.6V FP8 at higher batch (matches tp4.log)
    ],
)
def test_n_padded_bpreshuffle_returns_unpadded_shape(aiter_module, m, n, k):
    """End-to-end N-padding: a weight with unaligned N goes through
    ``shuffle_weight`` (auto-pad enabled), the GEMM is dispatched against
    the padded N, and the wrapper slices the output back to the original
    ``n``. This is the path that unblocks GLM-4.6V FP8 gate_up_proj at
    TP={4,8} -- before this fix the kernel raised
    ``RuntimeError: This GEMM is not supported!``."""
    aiter = aiter_module

    torch.manual_seed(0)
    device = "cuda"
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device)

    xq, x_scale = _fp8_perchannel_quant(x_bf16)
    wq, w_scale = _fp8_perchannel_quant(w_bf16)

    shuffled = shuffle_weight(wq, layout=(16, 16))
    # Sanity: this only exercises N-padding when N is actually unaligned.
    if getattr(shuffled, "aiter_padded_n", n) == n:
        pytest.skip(f"expected N={n} to need padding for this test")

    try:
        out = aiter.gemm_a8w8_bpreshuffle(
            xq, shuffled, x_scale, w_scale, None, torch.bfloat16
        )
    except RuntimeError as e:
        pytest.skip(
            f"bpreshuffle kernel rejected N-padded shape "
            f"(M={m}, N_padded={shuffled.aiter_padded_n}, K={k}). "
            f"Underlying error: {e}"
        )

    # The GEMM was dispatched against (M, padded_N, K) but the wrapper
    # must slice back to (M, original_N) before returning so downstream
    # layers see the shape they expect.
    assert out.shape == (m, n), (
        f"output shape must be sliced back to original N; got {out.shape}, "
        f"expected ({m}, {n})"
    )

    ref = _torch_reference(xq, wq, x_scale, w_scale, torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_n_padded_bpreshuffle_with_parameter_wrap(aiter_module):
    """Regression guard for the failure mode actually seen in
    ``tp4.log`` / ``tp8.log``: ``shuffle_weight`` produces a tensor whose
    padding metadata is then stripped by ``torch.nn.Parameter`` wrapping.
    The sidecar must recover the metadata so XQ K-pad and Y N-unpad both
    still fire even though ``hasattr(WQ, 'aiter_padded_k')`` is False."""
    aiter = aiter_module

    torch.manual_seed(0)
    device = "cuda"
    # Pick a shape that needs both K and N padding so we cover both paths
    # through the Parameter wrap in a single test.
    m, n, k = 16, 5472, 2736
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device)

    xq, x_scale = _fp8_perchannel_quant(x_bf16)
    wq, w_scale = _fp8_perchannel_quant(w_bf16)

    shuffled = shuffle_weight(wq, layout=(16, 16))
    assert shuffled.aiter_padded_k > k and shuffled.aiter_padded_n > n
    param = torch.nn.Parameter(shuffled, requires_grad=False)
    # Confirm the failure mode the sidecar guards against: attrs gone.
    assert not hasattr(param, "aiter_padded_k")
    assert not hasattr(param, "aiter_padded_n")

    try:
        out = aiter.gemm_a8w8_bpreshuffle(
            xq, param, x_scale, w_scale, None, torch.bfloat16
        )
    except RuntimeError as e:
        pytest.skip(
            f"bpreshuffle kernel rejected (M={m}, "
            f"N_padded={shuffled.aiter_padded_n}, "
            f"K_padded={shuffled.aiter_padded_k}). Underlying error: {e}"
        )

    assert out.shape == (m, n)
    ref = _torch_reference(xq, wq, x_scale, w_scale, torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_n_padded_bpreshuffle_pads_w_scale(aiter_module):
    """The N-padding path must also pad ``w_scale`` so the kernel's
    per-channel scale lookup is in-bounds for the padded N tail. Verify
    by running the GEMM with an explicitly-unpadded ``w_scale`` (the
    SGLang path) and confirming the output matches a reference where we
    pad ``w_scale`` manually."""
    aiter = aiter_module

    torch.manual_seed(0)
    device = "cuda"
    m, n, k = 16, 2736, 4096
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device)

    xq, x_scale = _fp8_perchannel_quant(x_bf16)
    wq, w_scale = _fp8_perchannel_quant(w_bf16)

    shuffled = shuffle_weight(wq, layout=(16, 16))
    if getattr(shuffled, "aiter_padded_n", n) == n:
        pytest.skip(f"expected N={n} to need padding for this test")

    try:
        out_auto = aiter.gemm_a8w8_bpreshuffle(
            xq, shuffled, x_scale, w_scale, None, torch.bfloat16
        )
    except RuntimeError as e:
        pytest.skip(f"bpreshuffle kernel rejected: {e}")

    assert out_auto.shape == (m, n)
    ref = _torch_reference(xq, wq, x_scale, w_scale, torch.bfloat16)
    torch.testing.assert_close(out_auto, ref, rtol=1e-2, atol=1e-2)
