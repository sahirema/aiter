#!/usr/bin/env python3
"""Prove the kernel's three write regions are disjoint and covering.

The Triton kernel does NOT fill sorted_ids with the padding value and then
overwrite the real slots, which is what aiter's reference does. Within a single
workgroup those two passes would need an intra-CTA barrier to be ordered, and a
missing fence there is the kind of bug that shows up as rare, shape-dependent
corruption on a node rather than as a clean failure. So it writes three regions
instead:

    R  real assignments        pos[t,e]                     for sel[t,e] == 1
    P  intra-expert padding    start[e] + counts[e] + r     for r < pad_n[e]
    T  tail                    s >= num_valid

If R, P, T are pairwise disjoint and their union is exactly [0, max_padded),
the order of the three passes cannot matter and no barrier is needed. If they
are not, the kernel is wrong. This checks both properties, and that the
assembled result still equals aiter's reference, on the same case space the
algorithm was validated over.
"""
import numpy as np
from algo_proto import ref_sort, router

BS = 32


def kernel_write_sets(sel, kslot, wsel, m, e, topk, block_size=BS):
    """Reproduce the Triton kernel's write pattern exactly, returning the three
    index sets alongside the assembled buffers."""
    max_padded = m * topk + e * block_size - topk
    max_blocks = (max_padded + block_size - 1) // block_size

    counts = sel.sum(0)
    blocks = (counts + block_size - 1) // block_size
    bstart = np.cumsum(blocks) - blocks
    start = bstart * block_size
    rank = np.cumsum(sel, axis=0) - sel
    pos = start[None, :] + rank
    total_blocks = int(blocks.sum())
    num_valid = total_blocks * block_size
    init_val = (topk << 24) | m

    ids = np.full(max_padded, -777, dtype=np.int64)
    wts = np.full(max_padded, np.nan, dtype=np.float64)
    eids = np.full(max_blocks, -777, dtype=np.int64)

    hit = sel == 1
    R = pos[hit]
    ids[R] = (kslot[hit].astype(np.int64) << 24) | np.nonzero(hit)[0]
    wts[R] = wsel[hit]

    P = []
    pad_n = blocks * block_size - counts
    for r in range(block_size):
        idx = np.nonzero(r < pad_n)[0]
        slots = start[idx] + counts[idx] + r
        P.append(slots)
        ids[slots] = init_val
        wts[slots] = 0.0
    P = np.concatenate(P) if P else np.array([], dtype=np.int64)

    T = np.nonzero(np.arange(max_padded) >= num_valid)[0]
    ids[T] = init_val
    wts[T] = 0.0

    # sorted_expert_ids: owners into [0, total_blocks), -1 into the rest
    Eo = []
    max_blk_per_e = (m * topk + block_size - 1) // block_size
    for b in range(max_blk_per_e):
        idx = np.nonzero(b < blocks)[0]
        Eo.append(bstart[idx] + b)
        eids[bstart[idx] + b] = idx
    Eo = np.concatenate(Eo) if Eo else np.array([], dtype=np.int64)
    Et = np.nonzero(np.arange(max_blocks) >= total_blocks)[0]
    eids[Et] = -1

    return dict(R=R, P=P, T=T, Eo=Eo, Et=Et, ids=ids, wts=wts, eids=eids,
                num_valid=np.array([num_valid, m]), max_padded=max_padded,
                max_blocks=max_blocks)


def check(m, e, topk, seed):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(m, e))
    bias = rng.normal(size=e)
    w, idx = router(scores, bias, topk)

    sel = np.zeros((m, e), dtype=np.int64)
    kslot = np.zeros((m, e), dtype=np.int64)
    wsel = np.zeros((m, e), dtype=np.float64)
    for t in range(m):
        for k in range(topk):
            sel[t, idx[t, k]] = 1
            kslot[t, idx[t, k]] = k
            wsel[t, idx[t, k]] = w[t, k]
    assert sel.sum() == m * topk, "a token selected an expert twice"

    K = kernel_write_sets(sel, kslot, wsel, m, e, topk)
    fails = []

    # 1. pairwise disjoint -- if violated, two passes race on the same slot
    for a, b in (("R", "P"), ("R", "T"), ("P", "T")):
        ov = np.intersect1d(K[a], K[b])
        if ov.size:
            fails.append(f"{a}/{b} overlap at {ov[:5].tolist()} ({ov.size} slots)")
    if np.intersect1d(K["Eo"], K["Et"]).size:
        fails.append("Eo/Et overlap")

    # 2. covering -- an uncovered slot keeps its sentinel and would be read as
    #    a live assignment by whatever ran before
    if np.unique(np.concatenate([K["R"], K["P"], K["T"]])).size != K["max_padded"]:
        fails.append(f"ids not covered: {K['max_padded'] - np.unique(np.concatenate([K['R'], K['P'], K['T']])).size} slots left")
    if np.unique(np.concatenate([K["Eo"], K["Et"]])).size != K["max_blocks"]:
        fails.append("expert_ids not covered")
    if (K["ids"] == -777).any() or (K["eids"] == -777).any():
        fails.append("sentinel survived a write pass")

    # 3. still equals aiter's reference
    r_ids, r_w, r_eids, r_nv = ref_sort(idx, w, e, BS)
    if not np.array_equal(K["ids"], r_ids):
        fails.append(f"sorted_ids differ at {np.nonzero(K['ids'] != r_ids)[0][:5].tolist()}")
    real = (r_ids & 0xFFFFFF) != m
    if not np.allclose(K["wts"][real], r_w[real], rtol=0, atol=0):
        fails.append("sorted_weights differ at real slots")
    if not np.array_equal(K["eids"], r_eids):
        fails.append("sorted_expert_ids differ")
    if not np.array_equal(K["num_valid"], r_nv):
        fails.append(f"num_valid {K['num_valid'].tolist()} vs {r_nv.tolist()}")
    return fails


def main():
    cases = [(18, 128, 4, s) for s in range(150)]
    for m in (1, 2, 7, 17, 18, 31, 32, 33, 64, 128):
        for e, topk in ((8, 2), (32, 4), (128, 4), (128, 8), (256, 1)):
            if topk > e:
                continue
            for s in range(4):
                cases.append((m, e, topk, 9000 + s))
    bad = 0
    for m, e, topk, s in cases:
        f = check(m, e, topk, s)
        if f:
            bad += 1
            print(f"FAIL m={m} e={e} topk={topk} seed={s}: {f}")
    print(f"\ncases: {len(cases)}   failures: {bad}")
    if bad:
        return 1
    print("ALL PASS - R/P/T are disjoint and covering, so the three write passes")
    print("           need no intra-CTA barrier, and the result still equals")
    print("           aiter's run_torch_moe_sorting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
