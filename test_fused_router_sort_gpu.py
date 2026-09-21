#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""GPU equivalence + benchmark for the fused router+sort kernel.

Run INSIDE the sglang container, on a gfx950 node:

    python test_fused_router_sort_gpu.py                 # real decode shape
    python test_fused_router_sort_gpu.py --sweep         # + shape sweep
    python test_fused_router_sort_gpu.py --bench         # + timings

Reference = the two kernels this replaces, composed:
    sglang.jit_kernel.moe_fused_gate.moe_fused_gate  ->  topk_weights, topk_ids
    aiter.fused_moe.moe_sorting                      ->  the 5-tuple

WHAT IS AND IS NOT DEFINED
    num_valid_ids bounds everything. aiter writes sorted_ids and
    sorted_expert_ids only up to it; the tail of both allocations is untouched
    torch.empty memory (verified on GPU 2026-09-20 -- the tail decodes to
    impossible ids such as k=-66, tok=11203349 on a fresh allocation, and to the
    previous case's ids once the caching allocator recycles the block). So the
    valid region is compared and the tail is reported only.
        Within the valid region sorted_ids IS fully defined -- real assignments
    and intra-block padding, the latter marked (topk << 24) | M -- and is
    compared in full. sorted_weights is torch.empty even at those intra-block
    padding slots (_moe_sorting_impl, fused_moe.py:205), so weights are compared
    only where a real token sits: slot < num_valid and (sorted_ids & 0xFFFFFF)
    != M.
        HISTORY: the first version of this file took the real-slot mask from the
    reference's ids across the WHOLE allocation. That classified uninitialised
    tail garbage as real and failed all 80 decode cases while the kernel was in
    fact bit-exact. A comparison is only as good as its definition of "defined".

Weights are compared by EXACT equality, not a tolerance. That is deliberate and
it is why the kernel reproduces sglang's routed_sum reduction along the K axis
rather than the N axis -- see the comment at that loop. A tolerance would hide
exactly the reduction-order drift worth knowing about. --atol relaxes it if the
peer wants to see how large a disagreement is rather than just that one exists.
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fused_router_sort import can_fuse, fused_router_sort  # noqa: E402

BLOCK_SIZE_M = 32          # aiter fused_moe.py:58
MODEL_DIM = 6144           # MiniMax-M3 hidden size
DECODE = dict(M=18, E=128, topk=4)   # measured decode shape, tp2 trace


def reference(scores, bias, topk, E, model_dim, moebuf_dtype, **gate_kw):
    from aiter.fused_moe import moe_sorting
    from sglang.jit_kernel.moe_fused_gate import moe_fused_gate

    topk_weights, topk_ids = moe_fused_gate(scores, bias, topk, **gate_kw)
    out = moe_sorting(
        topk_ids.to(torch.int32), topk_weights.to(torch.float32),
        E, model_dim, moebuf_dtype, block_size=BLOCK_SIZE_M,
    )
    # moe_sorting returns extra trailing values under some flag combinations;
    # the 5-tuple contract is the leading five (fused_moe.py:971 call site).
    return tuple(out)[:5] + (topk_weights, topk_ids)


def compare(ref, got, M, topk, atol, verbose):
    """Return (ok, lines). Splits ids into real vs padding so the two failure
    classes are never reported as one number."""
    r_ids, r_w, r_eids, r_nv, r_buf = ref[:5]
    g_ids, g_w, g_eids, g_nv, g_buf = got

    lines, ok = [], True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        lines.append(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    check("num_valid_ids", torch.equal(r_nv, g_nv), f"ref={r_nv.tolist()} got={g_nv.tolist()}")
    check("shapes", r_ids.shape == g_ids.shape and r_eids.shape == g_eids.shape,
          f"ids {tuple(r_ids.shape)}/{tuple(g_ids.shape)} eids {tuple(r_eids.shape)}/{tuple(g_eids.shape)}")
    if not ok:
        return ok, lines

    # The VALID REGION is what num_valid_ids names, and it is the only region
    # aiter writes at all. Everything at or beyond it is the untouched tail of a
    # torch.empty allocation -- measured 2026-09-20: it holds float garbage on a
    # fresh allocation (ids decoding to k=-66, tok=11203349) and stale ids from
    # the previous call once the caching allocator recycles the block. An earlier
    # version of this file identified padding as (ids & 0xFFFFFF) == M taken from
    # the reference, which silently classified that garbage as REAL and failed
    # 80/80 cases while the kernel was correct. Bound every check by num_valid.
    nv = int(r_nv[0])
    check("num_valid in range", 0 < nv <= r_ids.numel(), f"nv={nv} of {r_ids.numel()}")
    if not ok:
        return ok, lines

    # Inside the valid region aiter DOES write the marker, so ids are fully
    # defined there -- real assignments and intra-block padding alike.
    ids_bad = int((r_ids[:nv] != g_ids[:nv]).sum())
    check("sorted_ids (valid region)", ids_bad == 0, f"{ids_bad} of {nv} differ")

    real = torch.zeros_like(r_ids, dtype=torch.bool)
    real[:nv] = (r_ids[:nv] & 0xFFFFFF) != M
    n_real = int(real.sum())
    check("real-slot count", n_real == M * topk, f"{n_real} vs M*topk={M * topk}")

    # Weights are torch.empty even at intra-block padding slots, so they are
    # comparable only where a real token sits.
    dw = (r_w[real] - g_w[real]).abs()
    wmax = float(dw.max()) if n_real else 0.0
    check("sorted_weights (real slots)", wmax <= atol, f"max|diff|={wmax:.3e} atol={atol:g}")

    nblk = nv // BLOCK_SIZE_M
    eids_bad = int((r_eids[:nblk] != g_eids[:nblk]).sum())
    check("sorted_expert_ids (valid blocks)", eids_bad == 0,
          f"{eids_bad} of {nblk} differ")

    # Reported, never failed: beyond num_valid the reference is uninitialised, so
    # a difference here says nothing about this kernel. Both stage kernels stop
    # at the expert's token count and never read it.
    tail_bad = int((r_ids[nv:] != g_ids[nv:]).sum())
    lines.append(f"  info  beyond num_valid: {tail_bad} of {r_ids.numel() - nv} ids "
                 f"differ (reference is torch.empty there; not a defect)")

    nz = int((g_buf != 0).sum())
    check("moe_buf zeroed", nz == 0, f"{nz} non-zero elements")

    if verbose and not ok:
        bad = torch.nonzero(r_ids[:nv] != g_ids[:nv]).flatten()[:12].tolist()
        for s in bad:
            lines.append(f"    slot {s}: ref id={r_ids[s].item()} "
                         f"(k={r_ids[s].item() >> 24}, tok={r_ids[s].item() & 0xFFFFFF}) "
                         f"got id={g_ids[s].item()} "
                         f"(k={g_ids[s].item() >> 24}, tok={g_ids[s].item() & 0xFFFFFF})")
    return ok, lines


def one_case(M, E, topk, seed, dtype, atol, verbose, **gate_kw):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    scores = torch.randn(M, E, generator=g, device=dev, dtype=dtype)
    bias = torch.randn(E, generator=g, device=dev, dtype=torch.float32)

    ref = reference(scores, bias, topk, E, MODEL_DIM, torch.bfloat16, **gate_kw)
    got = fused_router_sort(
        scores, bias, topk, E, MODEL_DIM, torch.bfloat16,
        block_size=BLOCK_SIZE_M, **gate_kw,
    )
    return compare(ref, got, M, topk, atol, verbose)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--atol", type=float, default=0.0,
                    help="0 = exact equality (default); raise to size a disagreement")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU visible -- this test must run on a node", file=sys.stderr)
        return 2
    print(f"device: {torch.cuda.get_device_name(0)}")

    gate_kw = dict(scoring_func="sigmoid", renormalize=True,
                   routed_scaling_factor=1.0,
                   apply_routed_scaling_factor_on_output=True)

    cases = [(DECODE["M"], DECODE["E"], DECODE["topk"], s) for s in range(40)]
    if args.sweep:
        for M in (1, 2, 7, 17, 18, 31, 32, 33, 64, 128):
            for E, topk in ((8, 2), (32, 4), (128, 4), (128, 8), (256, 1)):
                cases.append((M, E, topk, 1000 + M * 17 + E + topk))

    failed = []
    for M, E, topk, seed in cases:
        ok_shape, why = can_fuse(M, E, topk)
        if not ok_shape:
            print(f"SKIP  M={M} E={E} topk={topk}: {why}")
            continue
        for dtype in (torch.bfloat16, torch.float32):
            ok, lines = one_case(M, E, topk, seed, dtype, args.atol,
                                 args.verbose, **gate_kw)
            if not ok:
                failed.append((M, E, topk, seed, str(dtype)))
                print(f"FAIL  M={M} E={E} topk={topk} seed={seed} {dtype}")
                print("\n".join(lines))
            elif args.verbose:
                print(f"ok    M={M} E={E} topk={topk} seed={seed} {dtype}")

    print(f"\ncases run: {len(cases) * 2}   failures: {len(failed)}")
    if failed:
        print("FAILED:", failed[:10])
        return 1
    print("ALL PASS - fused kernel matches sglang router + aiter moe_sorting")

    if args.bench:
        bench(gate_kw, args.iters)
    return 0


def bench(gate_kw, iters):
    from aiter.fused_moe import moe_sorting
    from sglang.jit_kernel.moe_fused_gate import moe_fused_gate

    M, E, topk = DECODE["M"], DECODE["E"], DECODE["topk"]
    dev = "cuda"
    scores = torch.randn(M, E, device=dev, dtype=torch.bfloat16)
    bias = torch.randn(E, device=dev, dtype=torch.float32)

    def unfused():
        w, i = moe_fused_gate(scores, bias, topk, **gate_kw)
        return moe_sorting(i.to(torch.int32), w.to(torch.float32), E, MODEL_DIM,
                           torch.bfloat16, block_size=BLOCK_SIZE_M)

    def fused():
        return fused_router_sort(scores, bias, topk, E, MODEL_DIM,
                                 torch.bfloat16, block_size=BLOCK_SIZE_M, **gate_kw)

    def timeit(fn):
        for _ in range(30):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000.0 / iters   # us

    # Back-to-back in a stream, so this measures the launch + dependency cost the
    # fusion targets -- NOT isolated kernel time. The trace attributes 7.00us to
    # the router and 9.62us to the sort per launch at this shape; a fused number
    # near or below their sum is the result to look for, and a number far above
    # it means the single-CTA sort became the bottleneck at this M.
    tu, tf = timeit(unfused), timeit(fused)
    print(f"\nM={M} E={E} topk={topk}, {iters} iters, back-to-back in-stream:")
    print(f"  router + moe_sorting : {tu:8.2f} us")
    print(f"  fused                : {tf:8.2f} us   ({tu / tf:.2f}x)")
    print("  (per-launch trace reference at this shape: 7.00 + 9.62 = 16.62 us)")


if __name__ == "__main__":
    raise SystemExit(main())
