# MiniMax-M3-MXFP4 GEMM tuning tables (MI355X / gfx950)

Tuned and untuned GEMM tables for `amd/MiniMax-M3-MXFP4`, captured and tuned on MI355X
(gfx950, `cu_num=256`), measured under SGLang v0.5.16.

**Bottom line: these tables produced no end-to-end gain on the machine they were tuned on,
and the one arm-pair that was repeated did not reproduce its own result.** They are published
so the measurement can be reproduced, or contradicted, on other hardware. Read
[Reading the numbers](#reading-the-numbers) before quoting anything from here.

## Files

| File | Rows | What it is |
|---|---|---|
| `minimax_m3_uplift_bf16_untuned_gemm.csv` | 771 | Dense bf16 shapes the runtime requests (aiter miss-dump census) |
| `minimax_m3_uplift_bf16_tuned_gemm.csv` | 741 | Tuner output for those shapes (`gemm_a16w16_tune.py`) |
| `minimax_m3_uplift_untuned_fmoe.csv` | 48 | Fused-MoE shapes requested, deduplicated across tp2/tp4/tp8 |
| `minimax_m3_tuned_fmoe.csv` | 27 | Tuner output for those shapes (`gemm_moe_tune.py`) |

TP is not part of either lookup key, so one file serves all TPs. The MoE table carries its TP
dependence through `inter_dim = 3072/tp` → 1536 (tp2), 768 (tp4), 384 (tp8).

## What these tables actually replace

**On the dense path the baseline is PyTorch, not a tuned aiter kernel.** Stock aiter ships 14
in-tree bf16 tables and 29 fmoe tables; measured overlap with the tables here is 0 rows on
both lookup keys. Zero overlap means the vendor tables do not *cover* these shapes — and the
arm logs confirm every miss falls through to `torch.matmul`:

| arm | dense misses | distinct (M,N,K,bias) | of which `using torch solution:0` |
|---|---|---|---|
| base tp2 | 3440 | 1732 | 100% |
| base tp4 | 124 | 44 | 100% |
| base tp8 | 17016 | 2143 | 100% |
| tuned tp2 / tp4 | 0 | 0 | — |
| tuned tp8 | 32 | 4 | 100% |

So for the shapes that moved, the A/B is **PyTorch vs the tuned kernel**.

**The vendor MoE table serves nothing in this deployment.** `minimax_m3_fp4_tuned_fmoe.csv`
is keyed `expert=129, topk=5`; this runtime requests `expert=128, topk=4`. 13 of the 15 key
columns agree and those two are disjoint, so it matches **0 of 48** requested shapes.
`minimax_m3_tuned_fmoe.csv` covers 27/48 — all 24 fp4-activation (prefill) shapes plus
`token=1` on the bf16-activation decode ladder. Decode at `token` 2–128 is heuristic-served
in both arms.

### Two traps when joining a request log to these tables

- **Miss-line counts are not call frequencies.** `get_GEMM_A16W16_config` is
  `@functools.lru_cache(maxsize=4096)` (`aiter/tuned_gemm.py:126`), so each distinct key logs
  at most once per process. tp4's 124 lines over 44 keys is roughly one per TP rank — a
  coverage census, not a hit rate. Nothing above says how often a shape runs.
- **M is not matched exactly.** The lookup tries exact M, then `get_padded_m(M,N,K,0)`, then
  `get_padded_m(M,N,K,1)` (`aiter/tuned_gemm.py:143`). `getPaddedM`
  (`csrc/py_itfs_cu/gemm_common.cu:13`) rounds up to a multiple of 16/32/64/128 by M band at
  `gl=0`, and to the next power of two at `gl=1`. M=324 is served by the M=352 row, M=540 by
  M=544. An exact-M join silently misses most rows.

Only misses are logged (`AITER_LOG_TUNED_CONFIG` gates the hit path), so these counts are a
lower bound on base-arm torch usage and say nothing about the hit shapes.

## Measured result

Eight-row VL image sweep (seeds 13–20), fp8_e4m3 KV, page-size 128, one run per cell, 3.4%
e2e noise floor. Each TP's base and tuned arms ran back-to-back on one node.

| TP | mean Δ | median Δ | worst row | mean, worst row dropped |
|---|---|---|---|---|
| tp2 | −5.23% | −2.22% | seed14 `mc=16`, −24.73% | −2.44% |
| tp4, run 1 | −8.29% | −4.06% | seed14 `mc=16`, −32.08% | −4.89% |
| tp4, run 2 | −1.71% | −0.13% | seed13 `mc=16`, −16.49% | −0.34% |
| tp8 | −4.49% | −3.08% | seed16 `mc=16`, −15.89% | −2.86% |

### Reading the numbers

**Do not quote a per-TP mean as the effect of these tables.**

1. Each mean is set largely by one row. Dropping each TP's worst row puts most of them inside
   the noise floor, and every median is milder than its mean.
2. The only arm-pair that was repeated did not reproduce. tp4 gave −8.29%, then −1.71% on a
   different node. Pooled row-by-row: **−5.00% (se 3.79, t = −1.32)** — not distinguishable
   from zero. The paired difference-of-differences between the two runs is **t = −1.59
   (df 7, p ≈ 0.15)**, so the two runs are not distinguishable from each other either. tp2
   and tp8 are n=1 on the same instrument and carry the same unmeasured uncertainty.

**Do not quote individual per-row gains either** — an unintervened control moves rows by
±10% (below).

### What survives replication: the concurrency structure

Pooled across all four arm-pairs (three TPs, tp4 contributing both runs):

| `max-concurrency` | n | median Δ | sign | two-sided sign test |
|---|---|---|---|---|
| 4 | 4 | −0.74% | 3/4 negative | p = 0.62 |
| 16 | 20 | **−7.25%** | **18/20 negative** | **p = 4×10⁻⁴** |
| 64 | 4 | +1.63% | 0/4 negative | p = 0.12 |
| 256 | 4 | +3.36% | 0/4 negative | p = 0.12 |

The re-run cost tp4 its mean but left this intact: 15/15 → 18/20 negative, median −7.84% →
−7.25%. All eight `mc=64`/`mc=256` cells are positive across four independent pairs.

The five `mc=16` rows within an arm-pair share a server process, so read the p-value as "the
sign is consistent across four separate base/tuned pairs on four nodes", not as twenty
independent trials.

### tp4, both runs

Same YAML, same arm-delta proof both times (`ARMPROOF=PASS`; tuned arm 0 dense misses, base
arm 124 misses / 124 torch fallbacks — identical kernel-selection signatures).

| seed | mc | run 1 Δ | run 2 Δ | mean |
|---|---|---|---|---|
| 13 | 16 | −27.73% | −16.49% | −22.11% |
| 14 | 16 | −32.08% | −9.08% | −20.58% |
| 15 | 16 | −1.79% | −6.28% | −4.04% |
| 16 | 16 | −6.32% | **+12.27%** | +2.97% |
| 17 | 16 | −9.98% | **+1.17%** | −4.40% |
| 18 | 4 | −0.95% | −0.53% | −0.74% |
| 19 | 64 | **+12.20%** | +0.27% | +6.23% |
| 20 | 256 | +0.37% | +4.99% | +2.68% |
| | | **−8.29%** | **−1.71%** | **−5.00% (se 3.79)** |

Two rows flipped sign. **Survives:** seeds 13 and 14 regress in both runs (−22%, −21%
averaged), and the `mc=16` sign pattern. **Does not:** the tp4 mean, and every individual row
magnitude.

Where the extra time went — benchmark seconds summed over the eight rows:

| | run 1 | run 2 | difference |
|---|---|---|---|
| base arm | 177.5 s | 178.0 s | −0.4 s |
| tuned arm | 193.1 s | 179.1 s | **+14.0 s** |

Run 1's tuned arm spent 14 extra seconds spread over 7 of its 8 rows — a uniform slowdown,
not one bad row. Generated-token counts agree within 2.3% on every row in both runs, so the
arms did the same work and the duration difference is real, not a generation-length artifact.

**Cause not established.** Every configuration-level explanation is eliminated: same YAML,
same merged tables, same fallback counts, same arm order. What remains is node state during
run 1, and the record there is partial. A chain-start `rocm-smi --showpids` shows the node
idle two seconds before the base arm launched — but the tuned arm started 46 minutes later
and nothing was captured in between, so a tenant arriving mid-chain is neither shown nor
excluded. One weak hint in that direction: run 1's tuned arm drifts +17.9% in per-batch
decode throughput from its first half to its second on `mc=16`, while the other three arms
are flat (+1.8%, −4.3%, +3.7%) — too sparse to carry a conclusion. Closing this needs
node-state capture **per arm**, not once per chain.

### How much a row moves with no intervention

The tp4 base arm was re-run unchanged on a second node 14 hours later — same YAML, identical
kernel-selection signature. Only the machine and the clock differ:

| seed | mc | node A | node B | Δ |
|---|---|---|---|---|
| 13 | 16 | 319.35 | 302.47 | −5.3% |
| 14 | 16 | 385.48 | 405.12 | +5.1% |
| 15 | 16 | 449.91 | 453.08 | +0.7% |
| 16 | 16 | 448.62 | 409.11 | −8.8% |
| 17 | 16 | 193.64 | 197.42 | +2.0% |
| 18 | 4 | 213.22 | 217.13 | +1.8% |
| 19 | 64 | 572.19 | 640.39 | **+11.9%** |
| 20 | 256 | 724.34 | 703.56 | −2.9% |
| | | | **mean +0.57%, median +1.27%** | |

- **Aggregates reproduce; single rows do not.** Two identical base arms agree to +0.57% on
  the mean, while the per-row spread is −8.8% to +11.9% — far wider than the 3.4% noise
  floor. The `mc=64` row moves +11.9% with no change at all, the same size as the +12.20%
  measured for tuning on that row. That is why individual row gains are not quotable.
- **Base-arm agreement does not validate the tuned delta.** It bounds noise on a base arm
  only. The `mc=16` contrast does hold: the control's `mc=16` rows are a coin flip (median
  +0.70%, 2/5 negative) against 18/20 negative for the tuned arms.

This is n=1 per row and confounds node identity with elapsed time. Treat it as proof that
±10% row swings happen unprovoked, not as a variance estimate.

**Confounded rows.** A row whose generated-token count moved between arms is confounded, not
faster or slower. Most drift <1.5%; tp8 seed17 moved −349 tokens (−6.5%) and tp4 seed17 +182
(+3.5%) — treat those two as confounded. tp4 seeds 13 and 14 drifted +0 and +10 tokens, so
the two largest regressions are not generation-length artifacts.

## Why: two hypotheses, neither settled

### H1 — splitK tuned on an idle GPU

Among shipped dense winners, splitK is selected almost exclusively at small M, the decode
regime:

| M bucket | n | winners with splitK>1 | median splitK |
|---|---|---|---|
| M≤8 | 39 | 87.2% | 6 |
| M 9–32 | 41 | 65.9% | 5 |
| M 33–256 | 142 | 28.9% | 4 |
| M>256 | 519 | 0.4% | 3 |

The tuner measures one GEMM at a time on an otherwise idle GPU, where splitK is close to
free: it recruits idle CUs and pays only a reduction pass. Under production decode the GPU is
not idle — attention, MoE and all-reduce are co-resident — so those CUs are contended, and
the reduction pass, its extra global traffic and the lost overlap are charged against TPOT.

This fits all three concurrency regimes: at `mc=4` the GPU is closest to the tuner's
conditions and the tables do no harm; at `mc=16` decode dominates and the splitK-heavy
small-M rows regress hardest; at `mc=64/256` throughput is dominated by large-M prefill
GEMMs, where splitK is ~0% of winners and the tuned choices are genuine wins.

How much those splitK kernels win by depends entirely on the comparison set. From the
profiling shards, for the 104 shipped rows selecting splitK>1 (medians of per-row ratios):

| M bucket | n | vs best splitK≤1, **any** libtype | vs best splitK≤1, **same** libtype |
|---|---|---|---|
| M≤8 | 34 | 1.235× | 1.620× |
| M 9–32 | 27 | 1.156× | 1.413× |
| M 33–256 | 41 | 1.198× | 1.676× |
| M>256 | 2 | 1.031× | 1.149× |
| **all** | **104** | **1.194×** | **1.617×** |

The same-libtype column is what capping splitK actually costs. This is reasoning over
measured selections, not an in-situ kernel comparison.

### H2 — it may be flydsl, not splitK

splitK and libtype are almost perfectly confounded at small M:

| M bucket | flydsl splitK>1 | flydsl splitK≤1 | triton (splitK≤1 by construction) |
|---|---|---|---|
| M≤8 | **34** | 1 | 4 |
| M 9–32 | 27 | 4 | 10 |

All 34 splitK>1 winners at M≤8 are flydsl. triton candidates carry `splitK=0`
unconditionally — `_get_triton_tasks` builds every candidate as
`info = (info_keys, 0, 0, "auto", "triton", is_shuffle)`, whose third element is splitK
(`csrc/gemm_a16w16/gemm_a16w16_tune.py:842`, unpacked at `:904`). So "splitK helps at small
M" and "flydsl wins at small M" are the same statement in this data. If flydsl's generated
decode kernels degrade under a contended GPU for unrelated reasons, every number in H1 looks
identical.

A second observation fits H2 and not H1: the TP that swapped the **fewest** shapes regressed
the **most** — tp4 replaced torch on 44 distinct shapes, tp8 on 2143. If harm scaled with
kernels swapped, that ordering should reverse. This is weak: it rests on n=1 for tp8, and on
tp4's re-run the two TPs are the same size in the opposite order.

### The ablation, and why it did not decide

A third tp4 arm ran the shipped table with every `splitK>1` entry replaced by the best
`splitK≤1` candidate **of the same libtype**, so the flydsl/triton mix is byte-identical and
splitK is the only variable (104 rows change; columns 1–11 of all 741 rows unchanged).
Predictions registered in advance, keyed on seeds 13/14 — the only rows with a reproducible
effect to remove: recovering toward 0 ⇒ H1; staying near −16.5%/−9.1% ⇒ H2; going further
negative ⇒ neither.

| seed | mc | tuned Δ | splitK-capped Δ | capped vs tuned |
|---|---|---|---|---|
| 13 | 16 | −16.5% | −17.4% | −1.1% |
| 14 | 16 | −9.1% | **+3.4%** | **+13.7%** |
| 15 | 16 | −6.3% | −5.4% | +0.9% |
| 16 | 16 | +12.3% | +14.1% | +1.6% |
| 17 | 16 | +1.2% | +0.3% | −0.9% |
| 18 | 4 | −0.5% | −1.6% | −1.1% |
| 19 | 64 | +0.3% | −1.7% | −2.0% |
| 20 | 256 | +5.0% | +3.9% | −1.1% |
| | | **mean −1.71%** | **mean −0.56%** | **mean +1.27%, median −0.96%** |

Benchmark seconds: base 178.0, tuned 179.1, capped 178.2 — all within 0.6%.

**The two discriminating rows disagree**: seed 13 did not recover (−1.1%), seed 14 recovered
fully (+13.7%). Both moves sit inside those rows' own run-to-run spread (11 pp and 23 pp).
**Neither hypothesis is confirmed or refuted.**

That is not only noise — the lever is structurally small, which bounds any repeat of this
experiment:

| | count |
|---|---|
| rows in the shipped table | 741 |
| rows the ablation changed | 104 |
| distinct shapes the tp4 runtime is confirmed to request | 44 |
| …resolving (after M-padding) to a changed row | **8** |

The 8 are `M=4` × {(1536, 24576, bias), (6144, 1536, ±bias)} at splitK 8/2/2, `M=16` ×
{(960, 1280, bias), (1280, 1280, ±bias), (1536, 1280, bias)} at splitK 4/4/5/5, and
`M=540→544` × (1536, 24576, bias) at splitK 4. The targeting is right — four are `M=16`
decode-batch shapes and `mc=16` is where the regression lives — but eight reachable rows
moved total benchmark time by 0.9 s out of 179. Both counts are **lower bounds**: the census
comes from the base arm's miss log, which sees only shapes the vendor tables fail to cover.

### Not refuted, and worth testing next

- **Selection at small M is a near-tie.** Against the second-best candidate of any libtype
  (`us>0`), the shipped winner leads by a median **0.62%** at M≤8 (n=35) and **0.25%** at
  M 9–32 (n=31), rising to **3.60%** at M>256 (n=518); 16 rows had no second candidate.
  Margins that small need not transfer from an idle GPU to a loaded one.
- **Numerics changed and are unvalidated**: `err_ratio` median 0.0124 at M≤8, max 0.0375
  table-wide, consistent with splitK reductions.

**Refuted:** that the tuner timed cache-resident weights — no shipped row reports bandwidth
above MI355X HBM peak (max 5.48 TB/s vs ~8 TB/s).

## Reproducing

Place the four CSVs in `aiter/configs/model_configs/`. aiter merges every table in that
directory at startup. Confirm from the `merge tuned file under` log line that both tuned
filenames appear, and check for zero `not found in a16w16 ... tune lookup table` errors.
Compare against a run with the two `*_tuned_*` files removed and the untuned ones left.

The per-candidate profile output (1.69M rows, 431 MB across 10 shards) is not committed; the
margin tables above were computed from it. The splitK-capped table is not committed either —
it is derived by replacing each `splitK>1` winner with the fastest `splitK≤1` row sharing its
10-column shape key *and its libtype*, leaving row order, row count and columns 1–11 intact.

Two things to get right when rebuilding it:

- **LF line endings.** Python's `csv.writer` defaults to `lineterminator="\r\n"`, which makes
  every line differ from the shipped table and turns a single-variable ablation into a
  whole-file change.
- **The canonical filename** `minimax_m3_uplift_bf16_tuned_gemm.csv`. aiter's merge globs
  `*bf16_tuned_gemm*.csv` and skips anything containing `untuned`
  (`aiter/jit/core.py:441-466`), so a table renamed `..._nosplitk_gemm.csv` is silently
  ignored and the arm benchmarks as if untuned. Distinguish arms by the directory you mount,
  not by the filename.

Measurement conditions worth repeating: run base and tuned back-to-back on one node; capture
node state (`rocm-smi --showpids`, `rocm-smi --showuse`, `docker ps`) **per arm** rather than
once per chain; and report generated-token counts per row so generation-length drift can be
excluded. One run per arm cannot resolve a sub-5% effect — plan at least two independent
arm-pairs per TP.
