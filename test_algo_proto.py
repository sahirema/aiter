"""Equivalence test: tile-shaped formulation vs aiter's reference sort.

Runs on CPU with numpy only -- no GPU needed. This validates the algorithm; the
Triton translation still needs a node.
"""
import sys
import numpy as np
from algo_proto import ref_sort, vec_sort, router

rng = np.random.default_rng(0)
fails = []


def check(name, topk_ids, topk_weights, E, bs):
    topk_ids = topk_ids.astype(np.int32)
    topk_weights = topk_weights.astype(np.float32)
    a = ref_sort(topk_ids, topk_weights, E, bs)
    b = vec_sort(topk_ids, topk_weights, E, bs)
    for field, x, y in zip(("sorted_ids", "sorted_weights", "sorted_expert_ids",
                            "num_valid"), a, b):
        if not np.array_equal(x, y):
            bad = np.flatnonzero(x != y)[:6]
            fails.append(f"{name}: {field} differs at {bad.tolist()} "
                         f"ref={x[bad].tolist()} vec={y[bad].tolist()}")


def distinct_topk(M, E, topk):
    """Each row picks `topk` DISTINCT experts, as the router guarantees."""
    return np.stack([rng.choice(E, size=topk, replace=False) for _ in range(M)])


# 1. the real decode shape, many random draws
for trial in range(200):
    M, E, topk, bs = 18, 128, 4, 32
    ids = distinct_topk(M, E, topk)
    check(f"decode/{trial}", ids, rng.random((M, topk)), E, bs)

# 2. shape sweep
for M in (1, 2, 7, 18, 31, 32, 33, 64, 128, 257):
    for E, topk in ((8, 2), (32, 4), (128, 4), (128, 8), (256, 1)):
        for bs in (16, 32, 64):
            if topk > E:
                continue
            ids = distinct_topk(M, E, topk)
            check(f"sweep/M{M}E{E}k{topk}b{bs}", ids, rng.random((M, topk)), E, bs)

# 3. adversarial: every token to the same expert set (long runs, many empty experts)
for M in (1, 33, 65, 128):
    E, topk, bs = 128, 4, 32
    ids = np.tile(np.array([0, 1, 2, 3]), (M, 1))
    check(f"degenerate/M{M}", ids, rng.random((M, topk)), E, bs)

# 4. exact block-boundary runs
for mult in (1, 2, 3):
    E, topk, bs = 64, 1, 32
    M = bs * mult
    ids = np.zeros((M, topk), np.int32)          # all on expert 0 -> exactly `mult` blocks
    check(f"boundary/x{mult}", ids, rng.random((M, topk)), E, bs)

# 5. last expert populated (tail of the expert axis)
E, topk, bs = 128, 4, 32
ids = np.tile(np.array([124, 125, 126, 127]), (18, 1))
check("tail-experts", ids, rng.random((18, topk)), E, bs)

# 6. scatter-collision guard: positions must be unique
for trial in range(50):
    M, E, topk, bs = 64, 128, 4, 32
    ids = distinct_topk(M, E, topk).astype(np.int32)
    w = rng.random((M, topk)).astype(np.float32)
    sel = np.zeros((M, E), np.int32)
    for k in range(topk):
        sel += (ids[:, k][:, None] == np.arange(E)[None, :])
    counts = sel.sum(0)
    blocks = (counts + bs - 1) // bs
    start = (np.cumsum(blocks) - blocks) * bs
    rank = np.cumsum(sel, 0) - sel
    pos = (start[None, :] + rank)[sel == 1]
    if len(np.unique(pos)) != pos.size:
        fails.append(f"collision/{trial}: {pos.size - len(np.unique(pos))} colliding slots")

# 7. router produces distinct experts per row (the assumption the sort rests on)
for trial in range(100):
    M, N, topk = 18, 128, 4
    sc = rng.standard_normal((M, N)).astype(np.float32) * 3.0
    bi = rng.standard_normal(N).astype(np.float32)
    _, idx = router(sc, bi, topk)
    for t in range(M):
        if len(set(idx[t].tolist())) != topk:
            fails.append(f"router/{trial}: row {t} has duplicate experts {idx[t].tolist()}")

# 8. router tie-breaking: identical scores must pick the lowest expert ids
sc = np.zeros((1, 16), np.float32)
bi = np.zeros(16, np.float32)
_, idx = router(sc, bi, 4)
if idx[0].tolist() != [0, 1, 2, 3]:
    fails.append(f"router-ties: got {idx[0].tolist()}, expected [0,1,2,3]")

print(f"cases run: decode=200 sweep+adversarial+boundary+tail, collision=50, router=101")
if fails:
    print(f"\nFAIL ({len(fails)}):")
    for f in fails[:15]:
        print("  " + f)
    sys.exit(1)
print("\nALL PASS - vec_sort is bit-identical to aiter's run_torch_moe_sorting reference")
