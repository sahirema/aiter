"""CPU prototype for the fused router+sort kernel.

Purpose: prove the *algorithm* the Triton kernel will implement, against aiter's
own reference, without needing a GPU. Only the Triton translation is then left
to validate on a node.

`ref_sort` is a faithful numpy transcription of
  container_src/aiter/op_tests/test_moe_sorting.py:44-97  (run_torch_moe_sorting)
which is the reference aiter validates its Opus sort against. `vec_sort` is the
tile-shaped formulation the kernel uses.

The load-bearing observation: a token never selects the same expert twice (the
router masks each winner to -inf before the next iteration), so an assignment's
rank within its expert is just the number of EARLIER TOKENS that chose it. That
collapses the rank computation from an [M*topk, E] problem to an [M, E] one,
which is what lets the whole sort live in a single workgroup at decode shapes.
"""
import numpy as np


def ref_sort(topk_ids, topk_weights, num_experts, block_size):
    """Transcription of aiter's run_torch_moe_sorting (the authority)."""
    m, topk = topk_ids.shape
    max_num_tokens_padded = topk_ids.size + num_experts * block_size - topk
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    init_val = (topk << 24) | m
    sorted_ids = np.full(max_num_tokens_padded, init_val, np.int32)
    # aiter leaves padding weights uninitialised (torch.empty); we write 0.0 so
    # the comparison is defined. Downstream masks on sorted_ids, not on weights.
    sorted_weights = np.zeros(max_num_tokens_padded, np.float32)
    sorted_expert_ids = np.full(max_num_m_blocks, -1, np.int32)
    num_valid = np.zeros(2, np.int32)

    b = eb = 0
    for e in range(num_experts):
        tok, kk = np.where(topk_ids == e)          # row-major: token asc, then k asc
        n = tok.size
        nb = (n + block_size - 1) // block_size
        sorted_ids[b:b + n] = (kk.astype(np.int32) << 24) | tok.astype(np.int32)
        sorted_weights[b:b + n] = topk_weights[tok, kk]
        b += nb * block_size
        sorted_expert_ids[eb:eb + nb] = e
        eb += nb
    num_valid[0] = b
    num_valid[1] = m
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid


def vec_sort(topk_ids, topk_weights, num_experts, block_size):
    """Tile-shaped formulation -- the exact sequence the Triton kernel runs."""
    m, topk = topk_ids.shape
    E = num_experts
    max_num_tokens_padded = topk_ids.size + E * block_size - topk
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    # [M, E] one-hot of the routing decision, plus the k-slot and weight carried
    # alongside. Built with a static loop over topk (topk is a constexpr, <= 8).
    sel = np.zeros((m, E), np.int32)
    kslot = np.zeros((m, E), np.int32)
    wsel = np.zeros((m, E), np.float32)
    ar_e = np.arange(E)[None, :]
    for k in range(topk):
        hit = (topk_ids[:, k][:, None] == ar_e)
        sel += hit
        kslot = np.where(hit, k, kslot)
        wsel = np.where(hit, topk_weights[:, k][:, None], wsel)
    assert sel.max() <= 1, "a token selected the same expert twice"

    counts = sel.sum(0)                                     # [E]
    blocks = (counts + block_size - 1) // block_size        # [E]
    bstart = np.cumsum(blocks) - blocks                     # [E] exclusive
    start = bstart * block_size                             # [E]
    rank = np.cumsum(sel, axis=0) - sel                     # [M,E] exclusive along tokens
    pos = start[None, :] + rank                             # [M,E]

    init_val = (topk << 24) | m
    sorted_ids = np.full(max_num_tokens_padded, init_val, np.int32)
    sorted_weights = np.zeros(max_num_tokens_padded, np.float32)
    t_idx = np.arange(m, dtype=np.int32)[:, None]
    hit = sel == 1
    sorted_ids[pos[hit]] = ((kslot << 24) | t_idx)[hit]
    sorted_weights[pos[hit]] = wsel[hit]

    # Block -> expert. The [bstart, bstart+blocks) intervals are disjoint and
    # cover [0, total_blocks), so exactly one expert matches each valid block and
    # the masked sum selects it.
    j = np.arange(max_num_m_blocks)[:, None]
    own = (bstart[None, :] <= j) & (j < (bstart + blocks)[None, :])
    sorted_expert_ids = np.where(
        own.any(1), (own * np.arange(E)[None, :]).sum(1), -1
    ).astype(np.int32)

    num_valid = np.array([blocks.sum() * block_size, m], np.int32)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid


def router(scores, bias, topk, routed_scaling_factor=1.0, renormalize=True,
           apply_scale=False):
    """sigmoid scoring + bias-ranked top-k, mirroring sglang's _router_triton_kernel
    (container_src/sglang/python/sglang/jit_kernel/moe_fused_gate.py:90-250) for
    SCORING_FUNC=0, N_GROUP=1, num_fused_shared_experts=0."""
    activated = 1.0 / (1.0 + np.exp(-scores.astype(np.float64)))
    biased = activated + bias[None, :]
    M, N = scores.shape
    idx = np.zeros((M, topk), np.int32)
    val = np.zeros((M, topk), np.float32)
    cur = biased.copy()
    for k in range(topk):
        win = cur.argmax(1)                      # lowest id wins ties (argmax semantics)
        idx[:, k] = win
        val[:, k] = activated[np.arange(M), win]
        cur[np.arange(M), win] = -np.inf
    if renormalize:
        s = val.sum(1, keepdims=True)
        val = val / np.where(s > 0, s, 1.0)
    if apply_scale:
        val = val * routed_scaling_factor
    return val.astype(np.float32), idx
