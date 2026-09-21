# Fused MoE router + sort

Replaces the pair

    _router_triton_kernel                         (sglang jit_kernel/moe_fused_gate.py)
    aiter::mxfp4_moe_sort_kernel<256,32,24,32>    (aiter moe_sorting)

with a single Triton kernel that also folds in the `moe_buf` zero-init.
Three kernels become one. **Default OFF behind `SGLANG_FUSE_MOE_ROUTER_SORT=1`.**

Scope is deliberately narrow: router + sort only. It does **not** touch
activation quant, which is what lets it sidestep both blockers that stop aiter
pr5517's `fused_moe_router` on the deployed stack — the `q_dtype_a` assert and
the missing `swiglu_limit` parameter (see
`traces/minimax-m3/ROUTER-FUSION-HANDOVER.md` §10.1, §10.4). Being
activation-agnostic is the whole point of this design.

---

## 1. Status — read this before quoting anything

| Claim | Evidence | Tier |
|---|---|---|
| Sort algorithm is bit-identical to aiter's `run_torch_moe_sorting` | `test_algo_proto.py`, ~700 cases, ALL PASS (CPU) | 1 |
| The kernel's three write regions are disjoint and covering | `test_write_regions.py`, 350 cases, 0 failures (CPU) | 1 |
| Both patches apply cleanly and the results parse | appliers run against copies; all 5+6 sites; `ast.parse` OK | 1 |
| The Triton kernel compiles | `probe_8b.py` on MI355X, 2026-09-20: `COMPILED AND RAN` | 1 |
| The Triton kernel is numerically correct on GPU **at E=256, topk=8** | same probe: 128 decoded assignments == torch reference, `num_valid=[480,16]`, `moe_buf` all-zero | 1 |
| The sort algorithm is correct at the deployed geometry E=128, topk=4 | `test_algo_proto.py:34` 200 trials + sweep/degenerate/tail cases, all PASS (CPU) | 1 |
| The **Triton specialization** at E=128, topk=4 runs correctly on GPU | **YES — bit-exact, 100/100 cases.** Equivalence against `moe_fused_gate` → `moe_sorting` on MI355X, 2026-09-20: M ∈ {1,2,7,17,18,31,32,33,64,128} × bf16/fp32, plus 40 seeds at M=18. `sorted_ids`, `sorted_expert_ids`, `num_valid_ids` exactly equal; weights exactly equal (§1.5) | 1 |
| The fusion is reachable on MiniMax-M3 as deployed | `num_fused_shared_experts` resolves to 0 on ROCm — chain in §7.1 | 2 |
| The election precondition holds at runtime | server log, 2026-09-20 07:41:28 TP0: `Shared experts fusion currently requires CUDA devices. Shared experts fusion optimization is disabled.` | 1 |
| The patched server starts and loads weights | same run: clean load, reached decode CUDA-graph capture at 08:00:41 | 1 |
| The unfused fallback path survives graph capture | **fixed and confirmed**: full 2-row arm on `96af5efded3d` ran to `ITT_RC=0`, 2026-09-20 08:23 | 1 |
| **The fusion is faster on the decode path** | **YES, ~4.5%/step.** Decode step period 13.08 vs 13.70 ms, batch-matched, both ranks, confirmed by *two* independent controls (same-session flag-off, and cross-day base image) — §1.3d | 1 |
| That decode gain shows up **end to end** | **NO — not resolvable, and now understood why.** 512p +0.5% (below §10's floor); 2048p's wall metric has **48.4% self-variance between two runs of the same arm**, so no e2e delta is quotable on it (§1.4) | 1 |
| Launch count on the decode path drops | **YES** — 114 -> 57 launches per decode pass, exactly 2:1, both ranks (§1.3) | 1 |
| The +14.1% 2048p e2e figure is caused by the fusion | **NO — retired as run-to-run variance.** The repeat put arm B at +69.3% vs the same control, i.e. 48.4% away from its own first run, on identical work (`ok=96`, `gen=5526`). Median TPOT moves +1.7%; the "+15.3% TPOT" that suggested a decode cost was *mean* TPOT, tail-contaminated (§1.4) | 1 |
| The fused kernel actually executes at runtime (§8(c)) | **YES** — 228 launches/rank, 4x57 MoE layers, both TP ranks identical; §1.3 | 1 |
| Graph capture stays healthy with the fusion on | `hipGraphLaunch` 57 vs 56-58; decode batch identical (`1x15, 8x16, 1x3`, all `cuda graph: True`) — no eager fallback (§1.3c) | 1 |
| §8(d) negative control — the **flag**, not the image, gates the kernel | same image, `SGLANG_FUSE_MOE_ROUTER_SORT=0` -> `_fused_router_sort_kernel` **0** on both ranks, vs **228** with it on (§1.3a) | 1 |
| It fires on **100% of decode** MoE layers, 0% of prefill | 228 fused pair 2:1 with 456 decode `MoeFlatmmKernel`, all inside decode windows; 342 `_router_triton_kernel` all in prefill (§1.3) | 1 |

The first compile happened on crsuse2-m2m-191 (MI355X) on 2026-09-20 and
needed no Triton type/broadcast fixes. What it did surface was a *torch* problem
rather than a Triton one — see §5.2.

The timing claim is still unestablished, and the first e2e attempt is worth
recording accurately because an earlier version of this paragraph got it wrong.
It said the ITT client was "killed mid-weight-load". It was not: the client ran
to completion of weight load, and the *server* died at 08:01:21 during decode
CUDA-graph capture from the arity bug in §5.3. The misreading came from an ITT
log frozen at exactly 32768 bytes — `run_ab_arm.sh:29` pipes ITT through
`grep -v`, and `grep` block-buffers at 32 KB when stdout is not a tty. **A log
stalled at a power of two is a buffering artifact, not a dead run.** Verify with
`ps` on the node before concluding anything from log silence. No throughput
number exists yet; do not infer one from the 5.47% share below.

Sizing, not a baseline, and weaker than it first looks. On
`traces/minimax-m3/tp2/tp2-512p-rank0.trace.json.gz` the router+sort pair is
**5.47% of decode kernel time** — a share, computed after excluding a 679 ms
cross-rank stall artifact that inflates raw "kernel time" by 33%. Three separate
reasons not to lean on it harder than that:

- The file's image and config are **UNKNOWN** (`tp2/PROVENANCE.md:6`) — a
  *script control only*, never an A/B control. (An earlier version of this line
  also said its KV cache was bf16. That was **wrong** and is retracted; the
  cache is fp8. See `tp2/PROVENANCE.md` "RETRACTION" and the section after it.
  Unknown provenance alone is sufficient for the script-control conclusion, so
  the conclusion is unaffected.)
- These tp2 captures are valid for **presence/absence and cross-kernel ratios,
  not absolute durations** (protocol 9; `inference-testing-f6`, confirmed in
  `PROVENANCE.md:11-16`). The 5.47% is a ratio and survives that. The
  corresponding ~947.7 µs/step **absolute does not** — do not quote it.
- An earlier pass put this chain at 18.17%. That figure was almost entirely the
  stall artifact and is wrong; it may still be in circulation.

So: the opportunity is real and worth a kernel, but the *size* of the win is
unestablished until measured against a machine baseline per uplift protocol §3.

Both kernels are **latency-bound at decode**, not compute- or bandwidth-bound:
the router launches 9 single-warp CTAs onto 256 CUs, and the sort moves 221 KB
in 9.62 µs (~23 GB/s). The win is removing a launch and a dependent round-trip,
not doing less arithmetic. That is also why the win should be *larger* at low
batch and vanish at high batch — and why the fused path is bounded to M ≤ 128.

## 1.1 Timing arm, 2026-09-20 — seeds 13 and 15

> **Resolved by §1.4 — the 2048p "+14.1%" below is retired.** A seed-15 repeat
> of arm B, same node and image, produced **0.2802 wall s/req against run 1's
> 0.1889 — a 48.4% spread between two runs of the identical arm**, with
> identical `ok=96` and `gen=5526`. Median TPOT across all three seed-15 runs
> moves only +1.7%/+5.4% while P99 moves +241%: the metric is tail-dominated and
> not reproducible. A 14.1% difference is under a third of this row's own noise.
>
> Two earlier versions of this banner got the attribution wrong in opposite
> directions — first "prefill-heavy, so not the fusion", then "mean TPOT +15.3%,
> so it is in decode". Both are superseded by §1.4; the sequence is kept there
> because the error was statistical, not experimental.
>
> Do not quote +14.1% as anything. The numbers below are what was measured.

Arm `mi355x_sglang_tp2_mm_fusedrs`, MLflow run `677e248637504ca3afb32b77c94807be`,
image `minimax-m3-sglang:v0.5.16-fp8fuse-decodeguard-fusedrs` (`96af5efded3d`),
node crsuse2-m2m-191, job 157030, one ITT invocation (§6), `ITT_RC=0`.
Log: `uplift_work/logs/fusedrs_20260920-080937.log`.

**Control.** The only valid control is the same base image with the flag absent:
`r7_e2e_guard.log`, image `minimax-m3-sglang:v0.5.16-fp8fuse-decodeguard`,
2026-09-19 16:56, same two seeds. The stock-image run `gemmbench_base_tp2.log`
(`amdsiloai/sglang:...uplift-tuned1060shapes-28082026`) is **not** a control for
this arm — it differs by the whole fp8fuse+decodeguard base, and comparing
against it would attribute the base-image delta to the fusion.

| seed / res | metric | A: decodeguard | B: +fusedrs | delta |
|---|---|---|---|---|
| 13 / 512x512 | realized completed | 94 / 96 | 96 / 96 | |
| | benchmark duration (s) | 20.97 | 21.52 | |
| | output throughput (tok/s) | 259.55 | 257.84 | -0.7% |
| | generated tokens | 5442 | 5548 | |
| | **generated tok/request** | **57.9** | **57.8** | -0.2% |
| | **wall s/request** | **0.2231** | **0.2242** | **+0.5%** |
| | Mean TTFT (ms) | 0.00 | 0.00 | artifact, see below |
| | Mean TPOT (ms) | 61.33 | 61.69 | contaminated |
| 15 / 2048x1600 | realized completed | 95 / 96 | 96 / 96 | |
| | benchmark duration (s) | 15.72 | 18.13 | |
| | output throughput (tok/s) | 348.32 | 304.77 | -12.5% |
| | generated tokens | 5475 | 5526 | |
| | **generated tok/request** | **57.6** | **57.6** | 0.0% |
| | **wall s/request** | **0.1655** | **0.1889** | **+14.1%** |
| | Mean TTFT (ms) | 26.69 | 25.75 | |
| | Mean TPOT (ms) | 45.30 | 52.21 | |

**Normalize per request before reading any of this.** A did not complete every
request (94 and 95 of 96) while B completed all 96. Output throughput and
`Total generated tokens` are totals over whatever completed, so the differing
denominator alone moves the raw gen-token counts by +1.9% and +0.9%. Per
request they are 57.9 -> 57.8 and 57.6 -> 57.6, i.e. **unchanged**. Protocol
§14 confounding is therefore *not* triggered: the arm did not flip thinking
decisions or change realized output length.

**Reading.** Seed 13 is +0.5% per request, inside the §10 3.4% noise floor —
report as *not resolvable e2e*. Seed 15 is +14.1% per request, outside it — an
apparent **regression**, not a speedup.

> **Retracted by §1.4.** "Outside the floor" assumed §10's 3.4% applies to this
> row. It does not: the seed-15 repeat put arm B 48.4% away from its own first
> run on identical work, so this row's own noise is an order of magnitude larger
> than the figure being read against it. Seed 15 is **not resolvable e2e**
> either — for a different reason than seed 13.

**Four reasons that regression is not yet a finding:**

1. ~~**§8(c) reachability is unestablished.**~~ **Resolved — §1.3.** The fused
   kernel ran on 100% of decode MoE layers (228 launches/rank, 2:1 with 456
   decode `MoeFlatmmKernel`) and 0% of prefill. The rows are not measuring a
   dormant patch. But the measured decode delta is -0.4%, which is why the
   attribution of the e2e numbers below does not survive.
2. **One unreplicated pair.** §10's noise floor is an e2e figure, not a licence
   to call a single pair. A and B also ran on different days under different
   jobs; node identity for A is not confirmed.
3. **TTFT is a broken metric on this path** (see below), so TPOT cannot carry
   any of the argument.
4. **§12 no longer applies at the deployed geometry.** Written when it did;
   §1.5 validated the fused path bit-exact at E=128, topk=4 across the whole
   fusable M range. Numerics are *unchanged*, not merely unvalidated.

**TTFT/ITL zero artifact.** `Mean/Median/P99 TTFT` and every `ITL` percentile
read exactly `0.00` on a nondeterministic subset of rows, in unpatched base
images too. Across five prior logs the TTFT-zero and ITL-zero row counts match
exactly (3/3, 3/3, 2/2, 2/2, 1/1); in `gemmbench_base_tp2.log` seeds 16/17/18
are zero while 13/14/15/19/20 are not; seed 13 reads `0.00` in one run and
`33.59` in another on the same config. It co-occurs with
`Total generated tokens (retokenized): 0` and `Peak output token throughput`
equal to `max-concurrency` — the client records one chunk per request and has
no intermediate token timestamps. Because
`TPOT = (e2e - TTFT) / (n - 1)`, a zeroed TTFT folds prefill into TPOT:
baseline seed 13's 33.59 ms over ~57 tokens is 0.59 ms/token, and
61.11 + 0.59 = 61.70 against a measured 61.69. **Do not compare TTFT across
rows, and do not compare TPOT unless both rows have nonzero TTFT.**

---

## 1.2 The deployed expert geometry, and what `can_fuse` does with it

From `config.json` of `amd/MiniMax-M3-MXFP4`
(`/shared_nfs/huggingface/models--amd--MiniMax-M3-MXFP4/snapshots/229447a2d6.../config.json`):

    num_local_experts    = 128      (not 256)
    num_experts_per_tok  = 4        (not 8)
    scoring_func         = sigmoid
    use_routing_bias     = True
    n_shared_experts     = 1
    (no n_group / topk_group keys -> grouped routing is OFF)

Against `can_fuse` every gate passes **on decode** and none passes on prefill:

| gate | deployed value | verdict |
|---|---|---|
| `M > max_m` (128) | decode M=16 (concurrency 16) | pass |
| | prefill M=1280..1408 | **reject — by design** |
| `num_expert_group > 1` | 1 (no n_group key) | pass |
| `num_fused_shared_experts != 0` | 0 on ROCm (§7.1) | pass |
| `topk > 8` | 4 | pass |
| `topk > num_experts` | 4 vs 128 | pass |

So the fusion can only ever fire on decode steps. A profile trace that shows
`_router_triton_kernel` and `mxfp4_moe_sort_kernel` still present is **not**
evidence of failure — prefill is supposed to keep using them. The 8(c)
observable is the *appearance* of `_fused_router_sort_kernel`, plus a drop in
the unfused counts, not their disappearance.

**This gap is CLOSED — see §1.5.** It read, until 2026-09-20 09:53: *"What is
not covered is the Triton specialization at `E=128, topk=4` on GPU.
`num_experts` and `topk` are constexpr there, so that geometry is a distinct
compiled kernel from the one `probe_8b.py` validated, with different
rank-collapse packing widths. The residual risk is a codegen/packing bug that
CPU prototyping cannot see, not an algorithmic one."*

That risk was the right one to name, and it did not materialise. The fused
kernel is bit-exact against `moe_fused_gate` → `moe_sorting` at `E=128, topk=4`
across every M the fusion can fire at. The gap was closed with
`test_fused_router_sort_gpu.py`, not the suggested `probe_8b.py` re-run: the
former compares against aiter's and sglang's *own* kernels, where `probe_8b.py`
compares against a hand-written torch reference and *reports* rather than
asserts.

---

## 1.3 Profile arm, 2026-09-20 — §8(c) reachability and the decode kernel bucket

Run: `mi355x_sglang_tp2_mm_profile_fusedrs`, image
`minimax-m3-sglang:v0.5.16-fp8fuse-decodeguard-fusedrs`,
`SGLANG_FUSE_MOE_ROUTER_SORT=1`, one active row (seed 13, 512x512), `ITT_RC=0`,
96/96 requests. Traces:
`traces/minimax-m3/tp2_fusedrs/1789893873.6120245/`.

Control: `r7_prof_guard` on the unpatched `...-decodeguard` base, **its seed-13
row only** — that config has two active rows and writes one trace dir per row;
`1789836652` is 16:50:52 UTC, the first row. Comparing against the seed-15 dir
would have compared different work.

### (a) The fused path fires, and fires exactly where `can_fuse` says

| observable | TP-0 | TP-1 |
|---|---|---|
| `_fused_router_sort_kernel` | 228 | 228 |
| `_router_triton_kernel` | 342 | 342 |
| `mxfp4_moe_sort_kernel` | 513 | 513 |

Both ranks agree to the launch. 57 is the MoE layer count and every count is an
exact multiple of it: the fused kernel fired on **4 forward passes x 57 layers**.

The four fused clusters (gap > 5 ms) are `[57, 57, 57, 57]`. Cluster 0 spans
102 ms against ~13 ms for the rest — that is Triton JIT/autotune on first touch,
and it is the §9 ramp, discarded everywhere below.

Associating kernels with those windows settles which path each pass took:

| kernel | total | in decode windows | outside |
|---|---|---|---|
| `_fused_router_sort_kernel` | 228 | **228** | 0 |
| `MoeFlatmmKernel` (decode MoE GEMM) | 456 | **456** | 0 |
| `mfma_moe1_*` (prefill MoE GEMM) | 342 | 39 | 303 |
| `_router_triton_kernel` | 342 | 39 | 303 |

`MoeFlatmmKernel` is 456 = **2 x 228**, pairing exactly 2:1 with the fused
kernel and lying wholly inside the decode windows. So every decode MoE layer ran
the fused path and **zero decode layers ran the unfused router**. The 39
prefill-kernel hits "inside" are an artifact of the +-50 ms window padding
catching prefill adjacent to the 102 ms JIT cluster, not real decode.

This is §8(c) satisfied. **§8(d) is satisfied twice, by two different kinds of
control:**

| control | isolates | `_fused_router_sort_kernel` |
|---|---|---|
| flag=1, patched image (the arm) | — | **228** / rank |
| flag=0, **same patched image**, same session | the *flag* | **0** / rank |
| unpatched `decodeguard` base, round-7 traces | the *whole patch* | **0** (2337 / 8607 unfused) |

The same-image flag-off run is the stronger of the two: it rules out the
possibility that the kernel's presence is a property of the image build rather
than of `SGLANG_FUSE_MOE_ROUTER_SORT`. Both return the opposite answer from the
arm, which is what §8(d) requires.

### (b) The decode kernel bucket — neutral, but it measures the wrong thing

> Superseded as a verdict by **(d)**, which measures the *whole* decode step
> including graph-replayed work and finds -4.5%. Keep this section: it is
> what the launch-count change costs on the eager path, and its blind spot
> is the reason (d) exists.

Per decode pass, summing every kernel the fusion replaces (arm:
`_fused_router_sort_kernel`; control: `_router_triton_kernel` +
`mxfp4_moe_sort_kernel` + `opus_moe_sorting_entry` +
`fused_mx_quant_moe_sort_kernel`), ramp pass dropped:

| rank | arm us/pass | arm launches | ctrl us/pass | ctrl launches | delta |
|---|---|---|---|---|---|
| TP-0 | 662.2 | 57 | 645.9 | 114 | **+2.5%** |
| TP-1 | 648.9 | 57 | 670.1 | 114 | **-3.1%** |
| mean | 655.5 | 57 | 658.0 | 114 | **-0.4%** |

**The launch halving is real and exact** — 2 kernels per decode MoE layer become
1, on both ranks. **The GPU time saving is not.** The two ranks disagree in
sign on the same rows, so the true effect sits inside rank-to-rank scatter. The
fused Triton kernel costs about what the two kernels it replaces cost, one of
which (`mxfp4_moe_sort` / aiter `opus_moe_sorting`) is hand-tuned CK. Halving
launch count does not pay for itself at this scale.

> **Scope limit — read before quoting the table above.** These numbers cover
> **eager decode passes only**, and there are just 3 of them per rank after the
> ramp. With `profile-steps: 64`, the window holds ~6 prefill + **57
> graph-replayed decode steps** (`hipGraphLaunch` = 57) + ~4 eager decode
> passes. Kernels inside a replayed hipGraph are **not** individually recorded
> — if they were, 57 replays x 57 layers would appear as thousands of extra MoE
> launches, and they do not. So the decode steps that actually determine TPOT
> are invisible to this measurement. The fused kernel is inside those graphs
> (captured at startup with the flag on), but its cost there is unmeasured.
> "Neutral" here means neutral *on eager decode*, not neutral on decode.

(Control pass 3 is not an outlier: 342 anchors is three decode passes merged by
the 5 ms clustering gap, and 1930.7/3 = 643.6 us/pass, consistent with the
clean 648.9 and 649.7. It is counted as three passes above, not one.)

### (c) The 2048p regression: unexplained here, retired in §1.4

> **Superseded by §1.4.** The repeat showed arm B differing from *itself* by
> 48.4% on this row, and median TPOT moving only +1.7%. The figure is retired as
> variance. This section is kept because its exclusions still hold and are what
> made the variance explanation the remaining candidate — but its central
> argument, quoted below, rests on *mean* TPOT and does not survive §1.4.

§1.1 reported 2048p at +14.1% wall/request. The attribution to the fusion is
**not supported by this profile, but it is not refuted either.** Being precise
about which, because an earlier draft of this section got it wrong:

*The regression is in decode, not prefill.* Seed 15's TTFT is clean and
essentially unchanged (26.69 -> 25.75 ms, -3.5%) while TPOT moves 45.30 ->
52.21 ms (**+15.3%**). TTFT is prefill, TPOT is decode. So the cost landed in
decode — which is exactly where the fusion lives.
>
> **This inference is wrong, and §1.4 says why:** 45.30 -> 52.21 is *mean* TPOT,
> which a latency tail destroys (the same row's P99 TPOT moved +46.5%). Median
> TPOT moves **+1.7%**. There is no decode cost to locate. An earlier draft argued the
opposite from "2048x1600 is the more prefill-heavy row"; that reasoning ignored
the TTFT/TPOT split and was wrong. It is corrected here rather than deleted.

What the profile *does* rule out and what it cannot reach:

- **Ruled out: CUDA-graph fallback.** The obvious mechanism by which a cheap
  kernel change produces an expensive decode regression is breaking graph
  capture, forcing eager decode. It did not happen: `hipGraphLaunch` is **57
  (arm) vs 58 (control)** and eager decode passes are 4 (arm) vs 5-6 (control).
  The arm runs *fewer* eager passes, not more. Graph capture is healthy with
  the fusion on.
- **Ruled out by magnitude — and the sign is backwards.** The router+sort stage
  is ~656 us of a ~13 ms decode pass, about 5%. A +-3% move there is ~0.15% of
  decode, ~100x too small for +15.3% TPOT. That bucket is eager-only (see the
  scope limit in (b)), so on its own it would be weak evidence — but **(d)
  closes that hole**: the inter-`hipGraphLaunch` decode step period, which
  *does* include graph-replayed work, is **-4.5%** with the fusion on. So the
  full decode step, measured end to end on the GPU, got *faster*. For the
  fusion to be the cause of +15.3% TPOT, it would have to slow decode by an
  amount the only unblinded decode measurement says it speeds up.
- **Not reachable: the profiled row is the wrong one.** This profile arm ran
  **seed 13 (512x512) only**. Seed 13 showed +0.6% TPOT — no regression. The
  regressing row, seed 15, was never profiled. Nothing here measures it.

So: the two mechanisms that could explain it are excluded, the profiled row is
not the regressing row, and the claim stands as **unexplained**. Do not quote
+14.1% as a fusion result, and do not quote it as refuted. The experiment that
would settle it is a profile arm on seed 15, flag on and off.


**Two mechanisms checked afterwards and refuted, from the timing logs:**

*Memory pressure is not it.* The candidate was that the fused kernel's workspace
shrinks the KV pool, which would bite only at 2048x1600 where per-request KV is
large — a mechanism that is resolution-dependent, as any explanation of this
result must be. It is dead on two independent counts: `max_total_num_tokens` is
**2143104 in both arms, identical**, and the scheduler reports `token usage:
0.01` throughout, i.e. ~1% of the pool. There is no pressure to be sensitive to.
No retraction or preemption events occur in either log (the only `retract` /
`preempt` matches are the `server_args` echo itself).

*Server-side decode telemetry points the other way.* Splitting the `Decode batch`
lines by row — the seed-15 block is the one with `#token` ~26k against seed 13's
~22k, the larger images — gives, with the ramp line dropped:

| seed-15 decode block (TP0) | A: decodeguard | B: +fusedrs |
|---|---|---|
| median `gen throughput` tok/s | 333.26 | 395.39 (**+18.6%**) |
| wall span of the block | 17 s (11 lines) | 14 s (10 lines) |
| `token usage` | 0.01 | 0.01 |
| `cuda graph` | True throughout | True throughout |

Both sub-signals say arm B's seed-15 *decode* was **faster**, while the client
reported it 15% slower end to end. This is a coarse instrument — 1-second
timestamps, ~10 samples, and the 1300+ tok/s entries are interval-boundary
artifacts — so it does not by itself overturn the client number. But it is a
third independent measurement agreeing with (d) and with the magnitude argument,
and disagreeing with the e2e figure.

(An earlier pass discarded this metric as contaminated, correctly at the time:
the medians were taken over all 20 lines, which span *both* seed rows. Splitting
by row removes that contamination and is what makes the comparison legitimate.)

So three independent measurements — decode step period, the kernel bucket, and
server-side gen throughput — say the fusion does not slow decode, while one
client-side measurement says decode got 15.3% slower. The weight of evidence
favours the e2e number being the anomaly, but that is a prior, not a result: it
is still n=1 per arm.

What remains unexplained is the 2048p e2e number itself. It is not attributed
to the fusion; it is an open measurement question (single run per arm, no
repeat, and §10's 3.4% noise floor was established on e2e wall, not on the
image rows specifically). Establishing it would need a repeat of both arms on
that row. Do not quote +14.1% as a fusion result.

### (d) Decode step period — the metric that actually sees graph-replayed decode

The kernel bucket in (b) is blind to the 57 graph-replayed decode steps. There
is a metric that is not: **the interval between successive `hipGraphLaunch`
records is the decode step period**, and it covers the whole step including
replayed work.

It is sound here for three reasons, each checked rather than assumed:

- The 57 launches sit at a **median 13.10 ms** period, matching the measured
  eager decode pass span (12.92 / 13.46 / 13.32 ms). Graph and eager decode
  steps run at the same rate.
- Host-side `hipGraphLaunch` duration is only ~465 us median (38 ms total), so
  the period is dominated by GPU work, not launch cost.
- Both arms decode at **identical batch size**: the server logs
  `Decode batch, #running-req: 16, ..., cuda graph: True` in each. Period is
  batch-size sensitive, so this had to be equal — and it is. That log line is
  also direct confirmation that decode runs under CUDA graphs.

Inter-batch idle gaps (>3x median — prefill and queue waits, not decode steps)
are dropped; 53 of 57 intervals survive in every capture.

**Primary comparison — same image, same session, flag is the only difference.**
`mi355x_sglang_tp2_mm_profile_fusedrs_off`, 2026-09-20 09:04, `ITT_RC=0`, 96/96,
traces `tp2_fusedrs_off/1789895066.3382428/`:

| rank | flag=1 (fused) | flag=0 (control) | delta |
|---|---|---|---|
| TP-0 | 13.09 ms | 13.70 ms | **-4.41%** |
| TP-1 | 13.07 ms | 13.69 ms | **-4.55%** |

**Batch-matched exactly.** Both runs decode with an identical batch
distribution — `1x #running-req:15, 8x 16, 1x 3`, all `cuda graph: True`. Since
step period is batch-size sensitive, this had to match, and it does.

Corroborated by the independent cross-day comparison against the *unpatched
base* image (`r7_prof_guard`, 2026-09-19): 13.09 vs 13.72 ms (-4.6%) on TP-0
and 13.07 vs 13.72 ms (-4.7%) on TP-1. Two controls of different kinds — one
isolating the flag, one isolating the whole patch — land within 0.3% of each
other, on both ranks. That is four measurements agreeing.

So on the decode path the fusion **is** faster, by ~4.5% per step.

**What this is not.** It is a profile-derived metric, so §7 bars it from
becoming an e2e or throughput claim. It once stood in tension with the timing
arm's 2048p figure; **§1.4 resolved that tension** by showing the 2048p wall
metric has 48.4% self-variance and its median TPOT moves +1.7%. There is no
longer a "4.5% decode gain vs 14.1% e2e loss" paradox — only a decode gain that
does not surface e2e at this concurrency.

**Reconciling -4.6% here with +0.6% TPOT in §1.1.** Not a contradiction: seed 13
is a TTFT=0 artifact row, so prefill is folded into its reported TPOT and that
TPOT is not a clean decode measure. The two numbers are measuring different
things. Seed 15 — the regressing row — has clean TTFT, and was not profiled.

### (e) What this arm does and does not license

**Licensed (Tier 1, both ranks):**
- The fusion is reachable and fires on **100% of decode MoE layers, 0% of
  prefill** — 228 launches/rank, pairing 2:1 with 456 decode `MoeFlatmmKernel`.
- It **halves decode router+sort launches**, 114 -> 57 per pass, exactly.
- **CUDA-graph capture stays healthy**: `hipGraphLaunch` 57 vs 56-58, identical
  decode batch distribution (`1x15, 8x16, 1x3`, all `cuda graph: True`). No
  eager fallback.
- **Decode step period -4.5%**, batch-matched, both ranks, both controls (d).
- §8(c) satisfied; §8(d) satisfied twice — same-image flag-off reads `fused=0`
  against the arm's 228, as does the unpatched base.

**Not licensed:**
- Any **throughput** number from this run (§7 — profile runs never supply
  timing).
- Any **e2e or throughput** claim. The -4.5% decode step period in (d) is a
  profile-derived metric; §7 bars profile runs from supplying timing. It says
  the decode *step* got faster, not that the benchmark did — and the timing arm
  disagrees at the e2e level, which is unresolved.
- Any claim that the fusion **caused** the 2048p regression, or that it did
  not. The profiled row is seed 13; seed 15 was never profiled (see (c)).
- Correctness at this geometry on GPU — still untested (§1.2).

~~**The two open experiments**~~ **Both closed — see §1.4.** (1) The seed-15
timing repeat ran and the +14.1% did **not** reproduce. (2) The seed-15 profile
pair is therefore **cancelled**: there is no effect left to locate.

### Pre-registered reading of experiment (1)

Written **before** the repeat's numbers were seen, so the interpretation cannot
be fitted to them afterwards. Comparator is the existing A row, unchanged and
unrepeated per the standing "never run ABBA" instruction: seed 15, `ok=95`,
`dur=15.72 s`, `out_tps=348.32`, `gen=5475`, TTFT `26.69`, TPOT `45.30`
(`logs/r7_e2e_guard.log`). Metric is **wall s/request**, i.e. benchmark duration
normalized by realized `completed`; A = 0.1655. B run 1 = 0.1889 (+14.1%).

| repeat B lands at | reading |
|---|---|
| ~0.185-0.193 (+12% to +17%) | +14.1% **reproduces**. Two independent runs agree, far above the 3.4% floor. The regression becomes a real, unexplained e2e effect that co-exists with a -4.5% decode step gain, and experiment (2) — a seed-15 profile pair — becomes mandatory. |
| ~0.164-0.171 (0% to +3.4%) | the first B was an **outlier**. The pair collapses into the noise floor, §10 says "not resolvable e2e", and the +14.1% figure is retired rather than explained. |
| anywhere between, or outside both | **neither** reading is licensed. A spread that wide across two runs of the identical arm means seed 15 is not reproducible at n=2 on this node, and the correct output is a variance statement, not a delta. |

The third row is the one that matters: it is pre-committed precisely so that a
messy middle result is not silently rounded toward whichever of the two clean
stories is more convenient. Note also that a *single* repeat of B against a
*single* A cannot separate "B is slow" from "that A was fast" — if the repeat
lands in row 1, the regression is established as reproducible for arm B, not yet
as a difference attributable to the fusion.

**Outcome: row 3 fired.** The repeat landed at **+69.3%** vs A — outside both
bands, and 48.4% away from arm B's own first run. Per the pre-registration the
licensed output is a variance statement, not a delta, and that is what §1.4
records. Worth noting that row 3 was written because a messy result seemed
*possible*; it turned out to be the actual result, and without it the honest
reading would have had to be argued for after the fact against two tidier
stories.


## 1.4 Seed-15 timing repeat, 2026-09-20 — the +14.1% does not reproduce

**Result: the 2048p regression is retired.** Not "unexplained" — explained, as
run-to-run variance in a statistic that is not reproducible on this row.

Arm B re-run alone on seed 15 (`rs_configs/fusedrs_arm_s15rep.yaml`, two lines
changed from `fusedrs_arm.yaml`: `run_name`, and the seed-13 row commented out).
Same node 157030, same image `...-decodeguard-fusedrs`, same
`SGLANG_FUSE_MOE_ROUTER_SORT=1`, `ITT_RC=0`, one ITT invocation (§6). Arm A was
not re-run, per the standing instruction. Log:
`uplift_work/logs/fusedrs_s15rep_20260920-090739.log`.

| seed-15 row | ok | gen | gen/req | wall s/req | vs A | **median** TPOT | vs A | **P99** TPOT | vs A |
|---|---|---|---|---|---|---|---|---|---|
| A: decodeguard | 95 | 5475 | 57.63 | 0.1655 | — | 46.41 | — | 67.01 | — |
| B: +fusedrs, run 1 | 96 | 5526 | 57.56 | 0.1889 | +14.1% | 47.22 | **+1.7%** | 98.15 | +46.5% |
| B: +fusedrs, repeat | 96 | 5526 | 57.56 | 0.2802 | **+69.3%** | 48.93 | **+5.4%** | 228.54 | +241.1% |

**The two B runs are the same arm on the same row on the same node, and they
differ by 48.4% in wall time** — while producing byte-identical work: `ok=96`
and `gen=5526` in both. Nothing about the workload moved (§14 satisfied:
gen/req -0.12% vs A). Only the clock did.

Spread of each statistic across all three runs:

| statistic | spread (max/min) |
|---|---|
| wall s/request | **+69.3%** |
| **median TPOT** | **+5.4%** |
| P99 TPOT | **+241.1%** |

That is the whole story. **The median decode step is stable to within 5.4%
across all three runs, while the tail moves 241% and wall time follows the
tail.** The seed-15 wall metric is tail-dominated, and the tail is not
reproducible. Against a B-to-B spread of 48.4%, a 14.1% A-to-B difference
carries no signal at all — it is under a third of this row's own noise, and
§10's 3.4% floor does not apply to a row that behaves like this.

### The correction this forces

§1.3(c) argued "the cost landed in decode" from **TPOT 45.30 -> 52.21 (+15.3%)**.
That is *mean* TPOT, and mean TPOT is precisely the statistic a latency tail
destroys — the same run's P99 moved +46.5%. The robust comparator is **median
TPOT, which moves +1.7%**, inside any reasonable floor. The decode-cost reading
does not survive the robust statistic.

This is the second time the seed-15 attribution has been revised, in opposite
directions, and both revisions came from reading a statistic more carefully
rather than from new data. Recorded plainly because the sequence is the finding:
first "prefill-heavy, so not the fusion" (wrong — ignored the TTFT/TPOT split),
then "TPOT +15.3%, so it is in decode" (wrong — mean TPOT, tail-contaminated),
now "median TPOT +1.7%, and B does not reproduce against itself."

Note also that all three seed-15 rows are touched by the TTFT/ITL zero artifact
to different degrees — `Median ITL` is 23.01 in A but **0.00 in both B runs**,
and the repeat's `Mean TTFT` is 0.00 outright. Since `TPOT = (e2e - TTFT)/(n-1)`,
a zeroed TTFT folds prefill into mean TPOT. That is an independent reason the
A-to-B *mean* TPOT comparison was never sound, and it was visible before the
repeat ran.

### What is now settled, and what the server said

The server was healthy in all three runs, which is why the variance is not a
serving failure: `cuda graph: True` throughout, `token usage: 0.01` (~1% of a
pool whose `max_total_num_tokens=2143104` is identical across arms),
**`#queue-req: 0` throughout**, and zero retraction or preemption events.

Combined with §1.3, five measurements now agree that the fusion does not slow
decode — decode step period (-4.5%), the eager kernel bucket (neutral),
server-side gen throughput, median TPOT (+1.7%), and the absence of any
scheduler pathology — and the single measurement that disagreed has been shown
not to reproduce against itself.

**Experiment (2) — the seed-15 profile pair — is cancelled, not deferred.**
There is no longer an effect for it to locate. Profiling a row whose wall metric
has 48% self-variance would only produce a trace of ordinary decode.

**What this does not license.** It does not make the fusion an e2e win. The
honest e2e statement is unchanged and is now better supported: **no resolvable
end-to-end effect on either row.** 512p was +0.5%, below §10's floor; 2048p is
not resolvable at all, because the row's own run-to-run spread swamps any delta
worth quoting. The decode-level gain in §1.3(d) is real and measured; it does
not show up e2e at this benchmark's concurrency, and this repeat explains why
the attempt to see it there produced a number rather than an answer.

## 1.5 GPU equivalence arm, 2026-09-20 — the deployed geometry is bit-exact

This closes the last open claim in §1. Run on `crsuse2-m2m-191` (job 157030),
image `96af5efded3d`, device **AMD Instinct MI355X**, via
`run_gpu_equiv.sh`. Logs: `logs/gpu_equiv{,2,3}_20260920-*.log`.

**Provenance.** The test file is mounted *alone* into an otherwise-empty
`/opt/frstest`, so its `sys.path.insert(0, <own dir>)` finds no local module and
the import falls through to the image's installed copy at
`/opt/venv/lib/python3.10/site-packages/fused_router_sort.py`. That copy hashes
`f87b819dbbbd4c94…`, identical to the working tree and to `fusedrs_build/`. This
validates the bytes the benchmark arm ran, not the working tree. aiter reported
`module_moe_sorting_opus.so`, i.e. the same Opus sorting backend the server
used — `AITER_USE_CK_MOE_SORTING` and `AITER_USE_FLYDSL_MOE_SORTING` are both
absent from the arm env and default to `"0"` (`aiter/fused_moe.py:62-63`).

### Result

| geometry | cases | result |
|---|---|---|
| **E=128, topk=4 (deployed)** | **100** — M ∈ {1,2,7,17,18,31,32,33,64,128} × {bf16, fp32}, plus 40 seeds at M=18 | **0 failures.** `sorted_ids`, `sorted_expert_ids`, `num_valid_ids` exactly equal; `sorted_weights` exactly equal; `moe_buf` all-zero |
| E=8 topk=2 / E=256 topk=1 | 40 | 0 failures |
| E=32, topk=4 | 20 | 14 fail — `sorted_weights` only, max\|diff\| ≤ 5.96e-08 |
| E=128, topk=8 | 20 | 8 fail — `sorted_weights` only, bf16 only, max\|diff\| ≤ 5.96e-08 |

M > 128 is rejected by `can_fuse`, so the 100 deployed-geometry cases span the
**entire range where the fusion can fire**. Decode-only run exits `rc=0`:
`cases run: 80  failures: 0  ALL PASS`.

**The 22 weight disagreements are ≤ 1 ULP and not at a deployed shape.** Every
one is `sorted_weights` alone — routing (`sorted_ids`, `sorted_expert_ids`,
`num_valid_ids`) is exact in all 180 cases, so no token is ever sent to the
wrong expert. The three distinct magnitudes are `1.49e-08`, `2.98e-08` and
`5.96e-08`: exactly 2⁻²⁶, 2⁻²⁵ and 2⁻²⁴, i.e. ¼, ½ and 1 ULP of fp32 at the
renormalized weight magnitude. This is the reduction-order drift the file's
exact-equality default exists to expose (see the `routed_sum` comment): the
reference's own tiling changes with N — `BLOCK_M = max(1, min(4, 256 //
BLOCK_N))`, `moe_fused_gate.py:313` — so N=32 and N=128 sum in different orders.
It is a property of the pair being compared, not a defect, and it does not occur
at `E=128, topk=4`.

### Microbenchmark — do not quote this as a decode saving

`--bench`, M=18 E=128 topk=4, 200 iters, back-to-back in-stream:

| | µs/call |
|---|---|
| `moe_fused_gate` + `moe_sorting` | 70.48 |
| fused | 26.52 (**2.66×**) |

This is **eager** launch-bound cost and it overstates the server's gain by a
wide margin: 57 layers × 44 µs would be 2.5 ms/step, against the 0.62 ms/step
actually measured in §1.3(d). The server replays decode inside a hipGraph, which
removes most of the host launch overhead this microbenchmark is dominated by.
**§1.3(d)'s 13.08 vs 13.70 ms/step (−4.5%) remains the only quotable decode
number.** The 2.66× is reported because it confirms the fusion removes real
launch/dependency cost, not because it sizes the deployment win.

### The harness defect this arm found, and why the first run said 80/80 FAIL

The first run (`gpu_equiv_20260920-094642.log`) failed **all 80** decode cases.
It was the test that was wrong, and the way it was wrong is worth keeping.

`compare()` took its real-slot mask from the reference's own ids across the
**whole** allocation — `real = (r_ids & 0xFFFFFF) != M` — on the file's stated
assumption that "sorted_ids, sorted_expert_ids, num_valid_ids and moe_buf ARE
fully defined". They are not. aiter writes only up to `num_valid_ids`; the tail
of both allocations is untouched `torch.empty` memory. The failure output showed
it directly: tail ids decoded to impossible values (`k=-66, tok=11203349`) on a
fresh allocation, and to *plausible* stale ids from the previous case
(`k=0, tok=3`) once the caching allocator recycled the block — which is why the
breakage looked like a routing bug rather than uninitialised memory.

Three structural identities held across all 80 failing cases and gave it away
before the fix:

- `real-slot count` reported 3694–3916 where only `M*topk = 72` can exist;
- `nreal − ids_bad == 72` **exactly, in all 80** — every slot the mask
  misclassified disagreed, and every genuinely real slot agreed;
- `sorted_expert_ids` differed in exactly `131 − num_valid/32` blocks **in all
  80** — i.e. precisely the blocks past `num_valid`, and none below it.

The fix bounds every check by `num_valid_ids`: ids are compared in full below it
(aiter *does* write the `(topk << 24) | M` marker there, for intra-block
padding), weights only at real slots below it, expert-ids only over
`num_valid // BLOCK_SIZE_M` blocks, and the tail is reported as `info` and never
failed. The pre-fix file is kept at `test_fused_router_sort_gpu.py.bak`.

The lesson is narrow and general: a comparison is only as good as its definition
of *defined*, and an exact-equality test over a `torch.empty` region
manufactures failures that say nothing about the kernel. The file's docstring
had reasoned carefully about this for `sorted_weights` and then asserted the
opposite for `sorted_ids` without checking — a confident comment is a claim made
at authoring time, not evidence.

### What this changes

**§12's "numerics changed, unvalidated" flag is lifted for this deployment.**
The fused path returns a bit-identical 5-tuple to the pair it replaces at
`E=128, topk=4` over the whole fusable M range, and `moe_buf` is zeroed
identically, so the downstream stages consume identical inputs. The arm does not
change model numerics as deployed. It is *not* lifted in general: at `E=32,
topk=4` and at `E=128, topk=8` in bf16 the renormalized weights drift by up to
1 ULP, which any future deployment at those geometries must re-validate.

---

## 2. Why it is correct

A token never selects the same expert twice: the router masks each winner to
`-inf` before the next iteration. So an assignment's rank within its expert
equals the number of *earlier tokens* that chose it, which collapses the rank
computation from `[M*topk, E]` to `[M, E]` — small enough that the entire sort
fits in one workgroup at decode shapes. `can_fuse` rejects `topk > num_experts`,
the only way that invariant can break.

The kernel does **not** fill-then-overwrite. That would need an intra-CTA
barrier for store ordering, and a missing fence there produces rare,
shape-dependent corruption that a smoke test will not catch. Instead it writes
three provably disjoint and covering regions — R (real scatter), P (intra-expert
padding, ≤ `BLOCK_SIZE_M−1` slots per expert), T (tail beyond `num_valid`).
`test_write_regions.py` is the proof, and it is the test to re-run first if you
change anything about slot assignment.

## 3. Files

    fused_router_sort.py          the kernel + can_fuse() + fused_router_sort()
    algo_proto.py                 CPU reference for the sort
    test_algo_proto.py            algo_proto vs aiter run_torch_moe_sorting
    test_write_regions.py         disjoint/covering proof  (CPU, no GPU needed)
    test_fused_router_sort_gpu.py equivalence + benchmark  (NEEDS A GPU)
    patches/apply_aiter_fused_router.py
    patches/apply_sglang_fused_router.py

`fused_router_sort.py` must be importable (`PYTHONPATH`) inside the server
container — both patches import it lazily by module name, and both treat
`ImportError` as "fall back", so a missing module degrades silently rather than
crashing. That means **a missing PYTHONPATH looks exactly like the flag being
off.** Prove reachability (§6) rather than trusting the flag.

## 4. Applying

Two independent patches, either order. Both are exact-match rather than unified
diffs — line numbers fail opaquely against a drifted tree, whereas a missing
anchor aborts loudly and writes nothing. Both are dry by default, refuse to
double-patch, and `ast.parse` the result before writing.

    python patches/apply_aiter_fused_router.py  <aiter>/aiter/fused_moe.py      # dry
    python patches/apply_aiter_fused_router.py  <aiter>/aiter/fused_moe.py --write

    python patches/apply_sglang_fused_router.py <sglang>/python/sglang          # dry
    python patches/apply_sglang_fused_router.py <sglang>/python/sglang --write

"sglang patched, aiter not" is a reachable state and is handled: the sglang side
probes `fused_moe`'s signature for a `fused_router` parameter and falls back
when absent. The reverse is inert — aiter's new parameter defaults to `None`.

Do **not** apply these to `uplift_work/container_src`. That is the extracted
reference copy of the shipped image; mutating it destroys the ability to diff
deployed vs. modified. Patch a build tree.

### What each patch does

**aiter** (6 sites, `fused_moe.py`): threads a `fused_router` parameter from
`fused_moe` → `fused_moe_` → `_fused_moe_impl`, then adds one `elif` branch at
the sort call site. Six sites and not one because neither `fused_moe` nor
`fused_moe_` takes `**kwargs` — both forward by explicit keyword, so a parameter
declared only on the outer signature is accepted and silently dropped.

The branch returns the same 5-tuple `moe_sorting` returns, so
`fused_moe_2stages` cannot tell the difference — it already accepts the sort
results as parameters and does not sort internally. That is the whole seam.

**sglang** (6 sites, 3 files): declares the env flag; elects
`TopKOutputFormat.BYPASSED` in `TopK.__init__`; handles the bypassed output in
`pre_permute_standard_to_aiter`.

No new seam was invented. `TopKConfig.output_format` is checked *first* in
`TopK.forward_cuda` (`topk.py:493`), ahead of every backend election, and
`BypassedTopKOutput` already carries router_logits + TopKConfig with a
`.to_standard()` that materialises routing the ordinary way.

The static/dynamic split matters: `output_format` is fixed per layer at
construction, but `can_fuse` depends on M, which changes every forward — a
decode batch fuses, a 512-token prefill must not. Electing statically and
falling back dynamically via `.to_standard()` is what lets one layer do both.

The election sits *after* the waterfill block in `TopK.__init__`, which forces
`STANDARD`. Waterfill must win, and by construction it does.

## 5. Fallback matrix

Every one of these returns to the unfused router + `moe_sorting` pair. None
raises.

| Condition | Checked in |
|---|---|
| flag off / not the aiter backend | `TopK.__init__` |
| grouped routing, fused shared experts, custom routing fn | `TopK.__init__` |
| no `correction_bias` (the kernel always reads one) | `TopK.__init__` |
| aiter wheel lacks `fused_router` | `_aiter_fused_moe_supports_fused_router` |
| `fused_router_sort` not importable | `_try_build_fused_router` |
| `apply_router_weight_on_input`, `expert_mask` (EP), `num_token_non_padded`, expert-location dispatch | `_try_build_fused_router` |
| router width ≠ `w13_weight.shape[0]` (redundant experts) | `_try_build_fused_router` |
| M > 128, topk > 8, topk > num_experts | `can_fuse` |
| `metadata.flat` sort layout, `need_local_topk_ids` | aiter branch (assert) |

The redundant-expert check earns its place: aiter derives its expert count from
`w1.shape[0]` and the kernel asserts `scores.shape[1] == bias.shape[0] ==` that
value, so a divergence would be an assert rather than a fallback. It is checked
where falling back is still possible.

## 5.1 Why this ships as patch scripts and not a fork branch

Decision, 2026-09-19: keep the patch scripts. Not a default — there is no
commit to branch from.

The deployed aiter records its own provenance in `aiter/_version.py`:

    __version__ = '0.1.19.dev109+ga63ede724.d20260805'

That is setuptools_scm. `ga63ede724` names commit `a63ede724`, and the trailing
`.d20260805` is its **dirty-tree** marker — the image was built from a working
tree with uncommitted changes. Measured against that commit, in
`/home/sahirema/aiter`:

| deployed `aiter/*.py` vs `a63ede724` | count |
|---|---|
| byte-identical | 556 |
| **different** | **8** |
| not present at that commit | 3 |

`fused_moe.py` is one of the eight and is 276 diff lines off. The delta
contains **`_moe_buf_or_alloc`** — the function this arm's `out_buf` handling
mirrors one-for-one. It does not exist at `a63ede724`.

So a branch based on that commit would not merely be untidy, it would be a
different base than the one the patch was written against: `BASE_SHA256`
(`f553937c…`) is the sha of the *deployed* file, and the guard would fail
immediately. Correctly. The deployed blob `73ca3f8cb…` is not in the fork's
object store at all, and the same holds on the sglang side — the deployed
`environ.py` blob is absent from that fork too.

The alternative considered and rejected: commit the deployed tree as a
synthetic base and stack the fusion on it. That buys a reviewable diff and a
PR, and 8 modified files is a tractable base — but it manufactures a commit
that exists nowhere upstream while looking authoritative. Revisit if the arm is
promoted and needs review; the delta is small enough to do cleanly then.

What the patch scripts already give that the branch was wanted for: the
sha256 guard fails loudly when the base moves, which is the property a pinned
branch provides. Protocol §11 asks for "a Dockerfile pinned to the base-image
commit" — here the base *is the image*, so `FROM <image>` + apply-patch is the
reproducibility artifact, with the patch script as its payload. Build that at
promotion, not now.

**Unrelated risk surfaced while establishing the above, recorded because it
affects every arm on this node:** those 8 modified files are in the image all
arms run against, and no git record of them exists anywhere. If that container
is rebuilt from a clean checkout, the delta — `_moe_buf_or_alloc` included —
disappears with no warning, and this arm's `out_buf` path loses the function it
was written to match.

## 5.2 Why the fused path bypasses the registered `fused_moe_` wrapper

First GPU contact did not fail in Triton. It failed at `import aiter.fused_moe`
with `AttributeError: 'types.UnionType' object has no attribute '__origin__'`,
raised from `torch.library.infer_schema` via aiter's
`jit/utils/torch_guard.py:239` while registering `fused_moe_` as a custom op.

Root cause, established empirically rather than assumed. PEP 604 unions are
*not* the problem: `torch.Tensor | None` hashes equal to `Optional[torch.Tensor]`
and is present in `torch._library.infer_schema.SUPPORTED_PARAM_TYPES`. The
problem is `tuple`, which is absent from that table in every spelling —
`tuple | None in S` is False and `Optional[tuple] in S` is False (52 entries
total). On a miss, torch reaches for `annotation_type.__origin__` to build its
"We do not support Tuple inputs in schema" diagnostic, and `types.UnionType`
has no `__origin__`, so the diagnostic itself raises. Switching to `Optional`
would only have converted the crash into that explicit error — there is no
spelling of a tuple parameter that a registered custom op accepts.

Three ways out were available:

| Option | Mechanism | Why not / why |
|---|---|---|
| A (**chosen**) | fused path calls `_fused_moe_impl` directly, so `fused_router` never appears on the registered `fused_moe_` | blast radius is the fused path only; `fused_moe_` is a *pure* forwarder (its whole body is that one call), so nothing is skipped |
| B | flatten the tuple into positional tensors | widens the registered signature for both arms and is a larger diff for no behavioural gain |
| C | aiter's `NONE_WRAPPED_OP` escape hatch (`torch_guard.py:200`) | **rejected on measurement grounds**: it de-registers `fused_moe_` for the *unfused baseline too*, contaminating the A/B control |

C is the one worth spelling out: it is the smallest patch and the most
tempting, and it is wrong precisely because it changes the arm we are measuring
*against*. The control must not move.

The consequence for the appliers: the aiter applier has **5** sites, not 6 —
there is deliberately no edit adding `fused_router` to the `fused_moe_`
signature. The two forwarding call sites have byte-identical tails, so they are
disambiguated by anchoring on surrounding unique text (the `result = fused_moe_(`
opener, and the trailing `# plumbing:` comment) rather than on the tails.
Patched sha256 of `aiter/fused_moe.py` is `9523d104b3c182d3f0e020cb2fb229a858bf7c8a49a89f2cc3a8aa5e06df8ed4`
(`7e949f70…` before the §5.3 fix, `116588e9…` before that; any Dockerfile
still pinning an older value is stale).


## 5.3 The 28th argument: option A's sharp edge, and the fix

Option A above splits the call by `fused_router is not None`:

```python
_impl = _fused_moe_impl if fused_router is not None else fused_moe_
```

The first cut then forwarded the new kwarg unconditionally into that one shared
call tail — `fused_router=fused_router,` — which is wrong for exactly the branch
option A was designed to protect. On the unfused branch `_impl` *is* the
registered custom op, whose schema has 27 parameters and no `fused_router`, so
passing it even as `None` is a hard dispatch error:

```
RuntimeError: aiter::fused_moe_() expected at most 27 argument(s) but received 28 argument(s).
  Declaration: aiter::fused_moe_(Tensor(a0!) hidden_states, Tensor(a1!) w1, ... )
Exception: Capture cuda graph failed: aiter::fused_moe_() expected at most 27 ...
```

Observed on MI355X, 2026-09-20 08:01:21, image `32ae4c4e60a2`: both TP ranks
died during decode CUDA-graph capture, ~20 min into a booked node, after a
clean weight load. The server exits 0 and `bench_serving` then sits in its
`--ready-check-timeout-sec 7200` window against a dead port — so the visible
symptom is a two-hour hang, not a crash. Look for `Scheduler hit an exception`
in the server log, not for a non-zero rc.

Two properties of this bug are worth internalising:

- **It lives on the fallback, not on the feature.** Every earlier GPU test
  exercised the fused path (`probe_8b.py` calls `fused_router_sort` directly)
  and so never touched it. The first thing a real server does is capture decode
  graphs at batch sizes where the fusion does not elect.
- **The count is the diagnosis.** 27 declared + 1 = 28. Counting the kwargs at
  the call site confirms the mechanism without needing a rerun.

The fix keeps the kwarg off the registered op entirely:

```python
_impl = _fused_moe_impl if fused_router is not None else fused_moe_
_fused_kw = {'fused_router': fused_router} if fused_router is not None else {}
result = _impl(..., out=out, residual=residual, **_fused_kw)
```

The unfused branch now passes `fused_moe_` exactly its 27 declared arguments —
byte-identical in behaviour to the base image, which is the property the whole
A/B rests on. Alternatives considered and rejected: making `_impl` always
`_fused_moe_impl` (de-registers the op for the arm's fallback path too, the
same control contamination that disqualified option C); and re-registering
`fused_moe_` with a widened schema (`gate_kwargs` is a dict, which has no
schema type either, so it does not actually close).

Fixed image ID `96af5efded3d`, built 2026-09-20 08:08 on crsuse2-m2m-191, all
§8(a) guards passing against the new `PATCHED_SHA`.


## 6. Testing on a node, in order

1. `python test_write_regions.py` — CPU, no GPU. Must stay 350/350.
2. `python test_fused_router_sort_gpu.py -v` — first compile. Equivalence
   against `moe_fused_gate` → `moe_sorting` at the decode shape
   (M=18, E=128, topk=4, model_dim=6144, BLOCK_SIZE_M=32), exact equality by
   default. **Every check is bounded by `num_valid_ids`**: aiter writes nothing
   past it, so the tail of `sorted_ids` and `sorted_expert_ids` is untouched
   `torch.empty` memory and is reported, never failed. Below `num_valid`, ids
   are compared in full (intra-block padding is genuinely marked
   `(topk << 24) | M` there) and weights only at real slots, where aiter's
   `torch.empty` weights actually hold a value. Taking that mask from the ids
   across the *whole* allocation is what made this test fail 80/80 on a correct
   kernel — §1.5.
3. `--sweep` for the shape sweep, `--bench` for the kernel-level timing. Note
   `--bench` only runs when every case passes (`main()` returns before it
   otherwise), and that `cases run` is `len(cases) * 2` regardless of `SKIP`s —
   so a run where `can_fuse` rejected everything still prints `ALL PASS`. Count
   the `ok` lines under `-v`; do not trust the summary alone.
4. Only then a server run with `SGLANG_FUSE_MOE_ROUTER_SORT=1`.

Proving the arm (uplift protocol §8, code arm): (a) provenance — build-time
guards on base sha256 of `aiter/fused_moe.py`, patched sha256 of the same file,
and AST presence of the four patched symbols (`fused_router_sort` imported by
aiter, `SGLANG_FUSE_MOE_ROUTER_SORT` assigned in `environ.py`,
`_try_build_fused_router` defined in `moe_runner/aiter.py`, `fused_router_sort`
+ `can_fuse` defined in the kernel module), checked **statically** (see "Where
each guard runs" below). Note it does *not* check for `fused_router` on the
`fused_moe_` signature — §5.2 explains why that parameter must not be there; (b) import identity
— `fused_router_sort.__file__` resolves inside the patched tree, checked on the
*imported* module; (c) reachability — the profile config
must show `aiter::mxfp4_moe_sort_kernel` **vanishing** and `_router_triton_kernel`
**vanishing**, replaced by one new kernel, with launches/step changing
accordingly. That absence/presence pair is the observable; a banner or an env
echo is not. (d) negative control: the same three assertions against the
baseline must return the other answer.

**Run this arm ONCE, against the baseline already on disk.** User instruction,
2026-09-19, relayed verbatim by `inference-testing-f6`, who stated the user
said it to them mid-turn: *"never run ABBA. A is alreayd run so you only ahv
eot run B"* (sic — quoted as relayed, typos included, because it is a relayed
quote and not something I should be tidying). The peer stated it applies to
this arm too. This is consistent with uplift protocol §3,
which already says the machine baseline is run once at the start of the round
with no baseline-last and no repeat — so there is no conflict to resolve, only
an ambiguity to close.

The ambiguity is in (d) above. "The same three assertions against the baseline"
does **not** license a second baseline run. (a) and (b) are build- and
import-time checks that run against the baseline *image* without any benchmark.
(c) needs a baseline *capture*, and the round's existing one is the one to use.
If anyone finds themselves scheduling a baseline run to satisfy (d), they have
misread it.

The cost of single-run is that capture identity stops being a controlled
blocking factor, so small deltas become unreadable. That cost is survivable for
*this arm's primary claim* because the predicted reachability effect is
categorical, not marginal: `_router_triton_kernel` and
`aiter::mxfp4_moe_sort_kernel` go from present to **absent**, replaced by one
new kernel. Capture drift cannot manufacture a kernel disappearing, so (c) and
(d) hold under a single pair.

It does **not** survive for the timing claim, and that asymmetry has to be
reported rather than glossed. A throughput delta from one unreplicated pair is
not resolvable against the 3.4% e2e noise floor (protocol §10). If this arm is
worth a timing number at all, the number comes from the separate timing run
against the existing machine baseline and is reported with its se per §10 — and
if it lands under the floor it is reported as "not resolvable e2e" with the
decode kernel buckets as the fallback, not as a speedup.

### Where each guard runs, and why (a) must not import aiter

Reported by `inference-testing-f6` from their decode_guard build (Tier 1, their
run): an import-identity check **cannot** run as a Dockerfile `RUN` stage.
Importing the sglang aiter backend pulls aiter, which probes the GPU during
import; `docker build` attaches no `/dev/kfd`, so it dies with
`RuntimeError: Get GPU arch from rocminfo failed`. They moved the check to
`docker run --device=/dev/dri --device=/dev/kfd`, and deliberately did *not*
stub `rocminfo` to make the build go green — a stub would verify the import
under an environment built to differ from the real one. That reasoning is right
and this arm follows it.

What that means here, checked against the deployed tree:

- **(a) provenance — static, build-time `RUN`, no GPU.** "`fused_router` present
  in the signature" means an **AST parse of the patched file on disk**, not
  `inspect.signature` on an imported module. Protocol §8(a) is already a static
  list (base commit, patch sha256, symbol present), so this costs nothing and
  removes the only reason (a) would need an import. Do not "simplify" it into
  an import later — that is what breaks the build.
- **(b) import identity — `docker run` with the devices attached.** For
  `fused_router_sort` itself the constraint is weaker than for their arm:
  it imports only `torch`, `triton`, `triton.language`
  (`fused_router_sort.py:50-52`) and never aiter, so that half would survive a
  build-time import. The *aiter/sglang* half does not, and (b) is only
  meaningful when it covers the symbol the runtime actually resolves. So the
  whole of (b) runs under `docker run`.

One correction to pass along, because it changes which knob is available. The
raise the peer saw is `chip_info.py:36`, but the import-time caller in
`jit/core.py:418` is `get_gfx_list()`, which **catches** `RuntimeError` and
falls back to `["cpu"]` (`chip_info.py:126-130`) — so that call is not the one
that kills the build. aiter splits the two deliberately, in its own docstrings:

- `get_gfx()` honours `GPU_ARCHS` and is documented for "build-time codegen
  paths … **where no GPU may be available**" (`chip_info.py:65-68`, `:123-135`).
- `get_gfx_runtime()` "always via rocminfo … **ignores GPU_ARCHS**", and calls
  `_detect_native()[0]` with no `try` (`chip_info.py:71-87`).

So `GPU_ARCHS=gfx950` is not a stub — it is aiter's own supported build-time
path — but it only disarms the `get_gfx()` family. Anything reaching
`get_gfx_runtime()` at import still raises. That makes `GPU_ARCHS` a live option
for build-time guards that must import aiter — but **not** for (b), where
running under the real device set is the point.

**Correction, and a caution about how the above was established.** An earlier
version of this section said "a package scan found import-time arch probes in
only three modules … none of them `get_gfx_runtime`". That was a *syntactic*
scan — it matched call sites whose callee name contained `gfx`/`arch`/`detect`
— and it was written as though it were an enumeration. It is not one.
`inference-testing-f6`'s traceback, verified here against the deployed tree,
names a path it could not have caught:

    ops/enum.py:19  ActivationType = type(_ActivationType(0))
      -> jit/core.py:1662  check_args()
      -> jit/core.py:433   get_asm_dir()  ->  get_gfx()  ->  _detect_native()

A module-level `_ActivationType(0)` reaches the probe through three frames and
matches no arch-shaped name. A second scan for import-time invocations of
`@compile_ops` symbols finds that shape in exactly one module —
`aiter/ops/enum.py:19-21` — so the gap is bounded. But two syntactic scans
catching two shapes is still not a transitive enumeration, and per the
"negative capability claims" rule, **no claim of the form "nothing else probes
at import" is established here.** The peer's traceback is the authority on what
actually fires; the scans only bound what was looked for.

That conclusion happens to be favourable — enum.py's chain runs through
`get_gfx()`, the `GPU_ARCHS`-honouring family, so the knob would have disarmed
it. Favourable is not the same as complete.

The same caution applies to a claim made just above about *this* arm, which
rests on the same weak method: `fused_router_sort.py:50-52` importing only
torch/triton is a read of its import lines, not proof that `import triton`
touches no device on ROCm. That has **not** been tested — there is no triton on
this host. Treat "that half would survive a build-time import" as unverified
reasoning, not a result. It does not affect the decision, because (b) runs
under `docker run` with devices regardless; it would matter only if someone
later moved a guard to build time on the strength of it.

Second hazard from the same report, recorded before this arm has a build script
so it is not reintroduced: a build step written as `docker build … | tail -60`
reports **rc=0 on a hard build failure**, because the pipeline's exit status is
`tail`'s. This arm currently ships no `.sh` and no Dockerfile, so it is clean
today. When one is added: `set -o pipefail`, or don't pipe the build at all.

Timing and profiling are separate runs (§7). Do not quote throughput from a
profile run.

## 7. Known limitations

- Ungrouped routing only. DeepSeek-style grouped routing (`num_expert_group>1`)
  is rejected, not silently mis-sorted.
- No expert parallelism (`expert_mask` / `num_local_tokens`).
- No fused shared experts.
- Requires a routing bias tensor. sglang handles the no-bias case elsewhere by
  allocating zeros per call (`topk.py:841-845`); that was deliberately *not*
  replicated here rather than add an untested path — models without routing bias
  keep the unfused path. MiniMax-M3 has `use_routing_bias=True` (nested under
  `text_config`), so the target model is unaffected.
- M ≤ 128. Prefill keeps the unfused path, by design.
- `topk ≤ 8` — `sorted_ids` packs the topk slot into the high 8 bits.
- **Renormalized weights can differ from the unfused pair by up to 1 ULP of
  fp32 at some non-deployed geometries** — measured at `E=32, topk=4` (both
  dtypes) and `E=128, topk=8` (bf16 only); max|diff| 5.96e-08. Routing is exact
  everywhere. The deployed `E=128, topk=4` is bit-exact across the whole fusable
  M range, so this does not affect MiniMax-M3 as deployed, but a deployment at
  another geometry must re-validate (§1.5).

## 7.1 Reachability on MiniMax-M3: why `num_fused_shared_experts` is 0

> **Supersedes the module docstring.** `fused_router_sort.py`'s SCOPE note cites
> "minimax_m3.py:1463 disables shared-expert fusion on ROCm". The substance is
> right but the citation has drifted (the function is now at 1481-1503) and the
> mechanism is a load-time override, not a direct assignment. The docstring is
> deliberately left unedited: the file is `COPY`d into the measured image, and
> changing it would alter the image ID and invalidate the §8(a) provenance
> record and the §8(b) GPU probe already run against `32ae4c4e60a2`. Fix it at
> the next rebuild; this section is authoritative until then.

"No fused shared experts" above reads like a blocker for this model and is not
one, but the reason is three files deep and was nearly misread — so it is
written out here rather than left to be rediscovered.

`text_config.n_shared_experts` is **1**, and both the election predicate
(`topk.py:452`) and `can_fuse` require `num_fused_shared_experts == 0`. Reading
only the MoE layer gives the wrong answer:

```python
# minimax_m3.py:293-297  -- inside MiniMaxM3MoE.__init__
self.num_fused_shared_experts = (
    0 if get_server_args().disable_shared_experts_fusion
    else config.n_shared_experts          # == 1
)
```

The flag defaults to `False` (`server_args.py:2081`) and nothing in the
canonical tp2 config sets it: `_moe_runner_fusion_disable` fires only for
flashinfer runners (we run `aiter`) and `_a2a_fusion_adjustments` only for
`deepep`/`megamoe`/`flashinfer` a2a backends (we set none). So on that reading
the value is 1 and the arm is a silent no-op.

What settles it is a *load-time override* one level up, in the top-level model:

| Step | Site | Result |
|---|---|---|
| 1 | runtime probe on the image | `is_cuda() == False`, `is_hip() == True` |
| 2 | `minimax_m3.py:1487-1488` | `disable_reason = "Shared experts fusion currently requires CUDA devices."` |
| 3 | `minimax_m3.py:1499-1503` | `declare_load_time_override(…, {"disable_shared_experts_fusion": True})` |
| 4 | `overrides.py:262-274` | "resolution has already materialized, so the declaration **writes through**" — calls `server_args.override(...)`, mutating the published object immediately |
| 5 | `minimax_m3.py:1456-1459` | that call runs **before** `MiniMaxM3Model(...)` is constructed |
| 6 | `minimax_m3.py:293-297` | therefore reads `True` → `num_fused_shared_experts = 0` |
| 7 | `minimax_m3.py:333-342` | `TopK(num_fused_shared_experts=0, top_k=4)` |
| 8 | `topk.py:452` | election condition satisfied |

Step 5 is the load-bearing one: the ordering, not the override, is what makes
this work. Reverse those two lines upstream and the fusion silently stops
electing with no error anywhere.

The runtime confirmation to look for in a server log is
`Shared experts fusion optimization is disabled.` — its absence means step 2
did not fire and the fusion will not elect.

**If this ever needs to work with the shared expert fused in** (a CUDA host, or
if upstream lifts the ROCm restriction), the extension is small and is worth
recording: with no deepep/megamoe the branch taken is `elif _aiter_append:` at
`topk.py:1957-1975`, which appends a single column of constant id `N` (=128) and
constant weight `scale_factor` — and `scale_factor` is `1.0` here because
`minimax_m3.py:333-342` never passes `fused_shared_experts_scaling_factor`. So
the kernel would emit slot 4 as `(id=128, weight=1.0)` and sort over 129
experts at `top_k=5`. No per-rank remap, no division, no gather. It is *not*
implemented — `can_fuse` still rejects `num_fused_shared_experts != 0`.
