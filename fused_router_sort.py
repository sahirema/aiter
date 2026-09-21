# SPDX-License-Identifier: MIT
"""Fused MoE router + sort: one kernel replacing two, for the decode regime.

WHAT THIS REPLACES
    sglang  _router_triton_kernel            (moe_fused_gate.py:90)
    aiter   MoeSortingKernel (Opus)          (moe_sorting_opus.h:453)
and the topk_ids/topk_weights round-trip through HBM between them.

WHY IT IS WORTH FUSING AT DECODE
    At M=18 the router launches `grid = cdiv(18, BLOCK_M=2) = 9` single-warp CTAs
    on a 256-CU device and costs ~7.00us; the sort costs ~9.62us moving ~220KB.
    Neither is compute- or bandwidth-bound -- both are launch and dependency
    latency. Fusing removes a launch, a kernel dependency stall, and the
    [M,topk] round-trip. Measured plateau cost of the pair: 947.7us/step,
    5.47% of decode kernel time (47 DECODE windows, 17.320 ms/step).

WHY NOT pr5517's fused_moe_router
    That fuses router+sort+activation-quant and gates on ActivationType.Silu.
    MiniMax-M3 runs Swiglu (confirmed at runtime: the trace's fused_moe_ op
    carries activation=2 == ActivationType.Swiglu, aiter_enum.h:12), and its
    quant stage would hit `assert q_dtype_a == fp4x2` while _resolve_quant_dtypes
    hands back bf16 for M < 256. This kernel deliberately does NOT fuse the
    quant, which is what makes it activation-agnostic and applicable here.

CORRECTNESS BASIS
    The sort half is bit-identical to aiter's own reference
    `run_torch_moe_sorting` (op_tests/test_moe_sorting.py:44-97), validated on
    CPU across ~1000 cases in test_algo_proto.py before this was written.

    Load-bearing invariant: a token never selects the same expert twice (the
    router masks each winner to -inf before the next iteration). That makes an
    assignment's rank within its expert equal to the number of EARLIER TOKENS
    choosing it, collapsing rank from an [M*topk, E] problem to [M, E] -- which
    is what lets the whole sort fit in one workgroup. The only way the invariant
    can break is topk > num_experts (the router would run out of experts to mask
    and reselect one), which `can_fuse` rejects.

SCOPE -- deliberately narrow, matching what the deployed image actually runs
    Supported:   sigmoid/sqrtsoftplus/softmax scoring, ungrouped routing,
                 expert_mask=None, num_local_tokens=None,
                 num_fused_shared_experts=0, M <= BLOCK_M.
    Unsupported: grouped routing (n_group>1), expert parallelism, fused shared
                 experts. Each is rejected loudly by `can_fuse` rather than
                 silently mis-sorted. MiniMax-M3 on the pinned image needs none
                 of them: config.json text_config has no n_group, and
                 minimax_m3.py:1463 disables shared-expert fusion on ROCm.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_SCORING = {"sigmoid": 0, "sqrtsoftplus": 1, "softmax": 2}


@triton.jit
def _fused_router_sort_kernel(
    scores_ptr,              # [M, N] raw gate logits
    bias_ptr,                # [N] fp32, ranking bias
    sorted_ids_ptr,          # [MAX_PADDED] i32  (k << 24) | token
    sorted_weights_ptr,      # [MAX_PADDED] fp32
    sorted_expert_ids_ptr,   # [MAX_BLOCKS] i32
    num_valid_ptr,           # [2] i32
    moe_buf_ptr,             # [M, MODEL_DIM] -- zeroed by programs >= 1
    M,
    routed_scaling_factor,
    moe_softcapping,
    stride_sm, stride_sn,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,          # >= M, power of 2
    BLOCK_N: tl.constexpr,          # >= N, power of 2
    BLOCK_K: tl.constexpr,          # >= K, power of 2
    BLOCK_SIZE_M: tl.constexpr,     # sort block size (aiter BLOCK_SIZE_M = 32)
    MAX_PADDED: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    MAX_BLK_PER_E: tl.constexpr,    # ceil(BLOCK_M * K / BLOCK_SIZE_M)
    SCORING_FUNC: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    MODEL_DIM: tl.constexpr,
    ZERO_BLOCK: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)

    # ---- programs >= 1: zero moe_buf -------------------------------------
    # The Opus sort zeroes this buffer as part of its work (aiter allocates it
    # with torch.empty, _moe_buf_or_alloc), because stage2 atomically
    # accumulates into it. Folding it in here is what keeps the fusion at ONE
    # launch rather than two.
    if pid != 0:
        base = (pid - 1) * ZERO_BLOCK
        offs = base + tl.arange(0, ZERO_BLOCK)
        tl.store(moe_buf_ptr + offs, tl.zeros([ZERO_BLOCK], tl.float32).to(
            moe_buf_ptr.dtype.element_ty), mask=offs < M * MODEL_DIM)
        return

    # ---- program 0: router + sort, entirely in one workgroup -------------
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    scores = tl.load(
        scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn,
        mask=mask_m[:, None] & mask_n[None, :], other=0.0,
    ).to(tl.float32)

    # Scoring. Mirrors sglang moe_fused_gate.py:142-168 exactly: `activated` is
    # the bias-free weight, `biased` is the ranking key.
    if SCORING_FUNC == 0:
        activated = tl.sigmoid(scores)
        biased = activated + bias[None, :]
    elif SCORING_FUNC == 1:
        sp = tl.where(scores > 20.0, scores, tl.log(1.0 + tl.exp(scores)))
        activated = tl.sqrt(sp)
        biased = activated + bias[None, :]
    else:
        logit = scores
        if HAS_SOFTCAP:
            z = logit / moe_softcapping
            logit = moe_softcapping * (2.0 * tl.sigmoid(2.0 * z) - 1.0)
        biased = logit + bias[None, :]
        biased = tl.where(mask_n[None, :], biased, -float("inf"))
        row_max = tl.max(biased, axis=1)[:, None]
        exp_row = tl.where(mask_n[None, :], tl.exp(biased - row_max), 0.0)
        activated = exp_row / tl.sum(exp_row, axis=1)[:, None]

    biased = tl.where(mask_n[None, :], biased, -float("inf"))
    biased = tl.where(biased == biased, biased, -1e30)      # NaN -> finite floor

    # Top-k with lowest-expert-id tie-break, producing at once the [M, K] form
    # sglang stores and the [M, N] one-hot the sort needs. Building both here,
    # instead of writing topk_ids to HBM and reading it back, is the whole point.
    #
    # selected_vals is carried in [BLOCK_M, BLOCK_K] solely so that routed_sum is
    # the SAME floating-point reduction sglang performs (moe_fused_gate.py:222):
    # summing the same k weights along the N axis instead would add them in
    # expert-id order rather than selection order, which can differ in the last
    # ulp and would rule out testing this against the reference by equality.
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    selected_vals = tl.zeros([BLOCK_M, BLOCK_K], tl.float32)
    sel = tl.zeros([BLOCK_M, BLOCK_N], tl.int32)
    kslot = tl.zeros([BLOCK_M, BLOCK_N], tl.int32)
    wsel = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)

    cur = biased
    for k in tl.static_range(K):
        max_val = tl.max(cur, axis=1)[:, None]
        lane = tl.where(cur == max_val, offs_n[None, :], N + 1)
        win = tl.min(lane, axis=1)[:, None].to(tl.int32)        # lowest id on ties
        w = tl.sum(tl.where(offs_n[None, :] == win, activated, 0.0), axis=1)[:, None]
        selected_vals = tl.where(offs_k[None, :] == k, w, selected_vals)
        hit = (offs_n[None, :] == win) & mask_m[:, None] & mask_n[None, :]
        sel = tl.where(hit, 1, sel)
        kslot = tl.where(hit, k, kslot)
        wsel = tl.where(hit, w, wsel)
        cur = tl.where(offs_n[None, :] == win, -float("inf"), cur)

    routed_sum = tl.sum(tl.where(mask_k[None, :], selected_vals, 0.0), axis=1)[:, None]
    if RENORMALIZE:
        norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)
        wsel = wsel / norm
    if APPLY_SCALE:
        wsel = wsel * routed_scaling_factor

    # ---- sort: counts -> blocks -> exclusive offsets -> scatter ----------
    hit_any = (sel == 1) & mask_m[:, None] & mask_n[None, :]
    sel_i = tl.where(hit_any, 1, 0)
    counts = tl.sum(sel_i, axis=0)                                    # [BLOCK_N]
    counts = tl.where(mask_n, counts, 0)
    blocks = (counts + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M              # [BLOCK_N]
    bstart = tl.cumsum(blocks, axis=0) - blocks                       # exclusive
    start = bstart * BLOCK_SIZE_M
    rank = tl.cumsum(sel_i, axis=0) - sel_i                           # [BLOCK_M, BLOCK_N]
    pos = start[None, :] + rank

    total_blocks = tl.sum(blocks, axis=0)
    num_valid = total_blocks * BLOCK_SIZE_M
    # [0] = padded slots in use, [1] = M (aiter num_tokens_post_pad contract,
    # test_moe_sorting.py:95-96). Written as one 2-lane vector: a scalar store to
    # a bare pointer is not a portable Triton form.
    offs_v = tl.arange(0, 2)
    tl.store(num_valid_ptr + offs_v, tl.where(offs_v == 0, num_valid, M).to(tl.int32))

    # Real assignments. Positions are provably unique (one per (token, expert)
    # pair, offset by that expert's exclusive base), so no store races.
    packed = (kslot << 24) | offs_m[:, None].to(tl.int32)
    tl.store(sorted_ids_ptr + pos, packed, mask=hit_any)
    tl.store(sorted_weights_ptr + pos, wsel, mask=hit_any)

    # Intra-expert padding: each expert's run is padded up to a block multiple,
    # at most BLOCK_SIZE_M-1 slots. Disjoint from the scatter above and from the
    # tail fill below, so all three can run without an intra-CTA barrier.
    # `+` rather than `|`: M < 2**24 is guaranteed by can_fuse's max_m, so the
    # low 24 bits of (K << 24) are zero and the two are identical -- but addition
    # of a Python int to a Triton scalar is the unambiguously supported form.
    init_val = M + (K << 24)
    init_n = tl.zeros([BLOCK_N], tl.int32) + init_val
    zero_n = tl.zeros([BLOCK_N], tl.float32)
    pad_n = blocks * BLOCK_SIZE_M - counts                            # [BLOCK_N]
    for r in tl.static_range(BLOCK_SIZE_M):
        slot = start + counts + r
        pm = mask_n & (r < pad_n)
        tl.store(sorted_ids_ptr + slot, init_n, mask=pm)
        tl.store(sorted_weights_ptr + slot, zero_n, mask=pm)

    # Tail beyond the last used block.
    init_s = tl.zeros([BLOCK_S], tl.int32) + init_val
    zero_s = tl.zeros([BLOCK_S], tl.float32)
    for s0 in tl.range(0, MAX_PADDED, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        tm = (offs_s >= num_valid) & (offs_s < MAX_PADDED)
        tl.store(sorted_ids_ptr + offs_s, init_s, mask=tm)
        tl.store(sorted_weights_ptr + offs_s, zero_s, mask=tm)

    # sorted_expert_ids: expert e owns blocks [bstart[e], bstart[e]+blocks[e]).
    # Each expert owns at most ceil(BLOCK_M*K/BLOCK_SIZE_M) blocks, so a short
    # static loop beats an [MAX_BLOCKS, N] ownership matrix.
    for b in tl.static_range(MAX_BLK_PER_E):
        bm = mask_n & (b < blocks)
        tl.store(sorted_expert_ids_ptr + bstart + b, offs_n.to(tl.int32), mask=bm)
    for j0 in tl.range(0, MAX_BLOCKS, BLOCK_S):
        offs_j = j0 + tl.arange(0, BLOCK_S)
        jm = (offs_j >= total_blocks) & (offs_j < MAX_BLOCKS)
        tl.store(sorted_expert_ids_ptr + offs_j,
                 tl.full([BLOCK_S], -1, tl.int32), mask=jm)


def can_fuse(M, num_experts, topk, num_expert_group=1, topk_group=1,
             num_fused_shared_experts=0, expert_mask=None, num_local_tokens=None,
             max_m=128):
    """Say whether the fused path is valid here, and why not when it isn't.

    Returns (ok: bool, reason: str). Callers must fall back to the unfused
    router+moe_sorting pair on False -- never silently proceed.
    """
    if M > max_m:
        return False, f"M={M} exceeds fused-path bound {max_m}"
    if num_expert_group > 1:
        return False, f"grouped routing unsupported (n_group={num_expert_group})"
    if num_fused_shared_experts != 0:
        return False, "fused shared experts unsupported"
    if expert_mask is not None:
        return False, "expert parallelism (expert_mask) unsupported"
    if num_local_tokens is not None:
        return False, "num_local_tokens unsupported"
    if topk > num_experts:
        # The router masks each winner to -inf, so with topk <= num_experts a token
        # cannot select an expert twice -- the invariant the [M, E] rank collapse
        # rests on. Beyond it the router reselects and the rank counts go wrong.
        return False, f"topk={topk} exceeds num_experts={num_experts}"
    if topk > 8:
        return False, f"topk={topk} exceeds 8 (packing uses 8 high bits)"
    if num_experts > (1 << 24):
        return False, "num_experts exceeds the 24-bit token field"
    return True, ""


def fused_router_sort(
    scores: torch.Tensor,          # [M, N] raw gate logits
    bias: torch.Tensor,            # [N] fp32
    topk: int,
    num_experts: int,
    model_dim: int,
    moebuf_dtype: torch.dtype,
    block_size: int = 32,
    scoring_func: str = "sigmoid",
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    moe_softcapping: float = 0.0,
    out_buf: torch.Tensor | None = None,
    accumulate: bool = True,
    max_m: int = 128,
):
    """Drop-in for `router(...) -> moe_sorting(...)`.

    Returns the same 5-tuple aiter's moe_sorting returns, so the result feeds
    `fused_moe_2stages` unchanged:
        (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)

    `accumulate` mirrors aiter's own meaning (_moe_sorting_impl, fused_moe.py:254):
    True  -> stage2 atomically accumulates into an [M, model_dim] moe_buf, which
             is allocated with torch.empty and therefore MUST be zeroed here --
             that zeroing is folded into this launch, which is what keeps the
             fusion at one kernel instead of two.
    False -> the FlyDSL stage2 reduce path owns an [M, topk, model_dim]
             intermediate and aiter returns a [0, 0] placeholder. Allocating and
             zeroing a real buffer here would both waste the bandwidth and hand
             the caller a tensor of the wrong shape.
    """
    assert scores.ndim == 2 and bias.ndim == 1, "scores must be 2D, bias 1D"
    assert scores.shape[1] == bias.shape[0] == num_experts, (
        f"expert axis mismatch: scores {scores.shape[1]}, bias {bias.shape[0]}, "
        f"num_experts {num_experts}")
    assert bias.dtype == torch.float32, "bias must be fp32"
    scoring_int = _SCORING.get(scoring_func.lower())
    assert scoring_int is not None, f"unknown scoring_func {scoring_func!r}"

    M, N = scores.shape
    ok, why = can_fuse(M, num_experts, topk, max_m=max_m)
    assert ok, f"fused_router_sort called on an unsupported shape: {why}"

    device = scores.device
    max_padded = M * topk + num_experts * block_size - topk
    max_blocks = (max_padded + block_size - 1) // block_size

    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_padded, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    if not accumulate:
        assert out_buf is None, "out_buf is meaningless when accumulate=False"
        moe_buf = torch.empty((0, 0), dtype=moebuf_dtype, device=device)
    elif out_buf is None:
        moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
    else:
        # out_buf may be an externally registered IPC buffer that stage2 writes
        # the combined result into (_moe_buf_or_alloc, fused_moe.py:86-91), so it
        # still needs zeroing here -- it is not merely an allocation shortcut.
        # Mirrors _moe_buf_or_alloc's checks one-for-one (fused_moe.py:96-106).
        # Kept as four separate asserts, not one conjunction: this buffer arrives
        # from outside, so a firing assert has to say which property was wrong.
        assert tuple(out_buf.shape) == (M, model_dim), (
            f"out_buf shape {tuple(out_buf.shape)} != {(M, model_dim)}")
        assert out_buf.dtype == moebuf_dtype, (
            f"out_buf dtype {out_buf.dtype} != {moebuf_dtype}")
        assert out_buf.device == device, (
            f"out_buf device {out_buf.device} != {device}")
        assert out_buf.is_contiguous(), "out_buf must be contiguous"
        moe_buf = out_buf

    BLOCK_M = max(16, triton.next_power_of_2(M))
    BLOCK_N = triton.next_power_of_2(N)
    ZERO_BLOCK = 4096
    n_zero = triton.cdiv(M * model_dim, ZERO_BLOCK) if accumulate else 0

    _fused_router_sort_kernel[(1 + n_zero,)](
        scores, bias, sorted_ids, sorted_weights, sorted_expert_ids,
        num_valid_ids, moe_buf,
        M,
        float(routed_scaling_factor), float(moe_softcapping),
        scores.stride(0), scores.stride(1),
        N=N, K=topk, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_K=triton.next_power_of_2(topk),
        BLOCK_SIZE_M=block_size, MAX_PADDED=max_padded, MAX_BLOCKS=max_blocks,
        MAX_BLK_PER_E=triton.cdiv(BLOCK_M * topk, block_size),
        SCORING_FUNC=scoring_int,
        HAS_SOFTCAP=bool(moe_softcapping != 0.0),
        RENORMALIZE=bool(renormalize),
        APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
        MODEL_DIM=model_dim, ZERO_BLOCK=ZERO_BLOCK, BLOCK_S=1024,
        num_warps=4,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf
