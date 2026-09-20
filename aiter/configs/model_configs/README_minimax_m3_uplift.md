# MiniMax-M3-MXFP4 GEMM tuning tables (MI355X / gfx950)

Tuned and untuned GEMM tables for `amd/MiniMax-M3-MXFP4`, captured and tuned on
MI355X (gfx950, `cu_num=256`). **These tables regressed the end-to-end benchmark on
the machine they were tuned on** — they are published so the result can be
reproduced, or contradicted, on different hardware.

## Files

| File | Rows | What it is |
|---|---|---|
| `minimax_m3_uplift_bf16_untuned_gemm.csv` | 771 | Dense bf16 GEMM shapes the runtime actually requests (aiter miss-dump census) |
| `minimax_m3_uplift_bf16_tuned_gemm.csv` | 741 | Tuner output for those shapes (`gemm_a16w16_tune.py`) |
| `minimax_m3_uplift_untuned_fmoe.csv` | 48 | Fused-MoE shapes the runtime requests, deduplicated across tp2/tp4/tp8 |
| `minimax_m3_tuned_fmoe.csv` | 27 | Tuner output for those shapes (`gemm_moe_tune.py`) |

TP does not appear in either lookup key, so one file serves all TPs. For the MoE
table the TP dependence is carried by `inter_dim = 3072/tp` → 1536 (tp2), 768 (tp4),
384 (tp8), all present in one file.

## Two things to know before comparing against these

1. **The baseline is not untuned.** Stock aiter already merges
   `minimax_m3_eagle_bf16_tuned_gemm.csv`, `minimax_m3_fp4_tuned_fmoe.csv`,
   `minimax_m3_mxfp8_tuned_fmoe.csv`, 14 in-tree bf16 tables and 29 in-tree fmoe
   tables. Measured overlap of these tables against that pool is **0 rows** on both
   the dense 10-column key and the fused-MoE 15-column key, so this is *tuned choice
   vs aiter's runtime heuristic on previously-uncovered shapes*, not a replacement of
   an existing tuned choice.
2. **The shipped `minimax_m3_fp4_tuned_fmoe.csv` is keyed `expert=129, topk=5`;
   this runtime requests `expert=128, topk=4`.** 13 of the 15 key columns agree;
   `expert` and `topk` are disjoint. Consequence: the vendor MoE table matches
   **0 of 48** requested shapes and serves nothing in this deployment.
   `minimax_m3_tuned_fmoe.csv` covers 27/48 — all 24 fp4-activation (prefill) shapes
   and `token=1` on the bf16-activation (decode) ladder. Decode at `token` 2–128
   is heuristic-served in both arms.

## Measured result (SGLang v0.5.16, MI355X, fp8_e4m3 KV, page-size 128)

Eight-row VL image sweep, seeds 13–20, one run per cell; 3.4% e2e noise floor.
Mean output-throughput delta, tuned vs base: **tp2 −5.23%, tp4 −8.29%, tp8 −4.49%**.
Reproduced on three separate nodes, which rules out node contention.

Losses concentrate at `max-concurrency=16` (worst: tp4 seed 14, −32.1% throughput /
+45.4% TPOT). `max-concurrency=4` is unaffected on every TP (+2.8 / −1.0 / −1.4%,
all inside noise). The only gains are at high concurrency: tp4 `mc=64` +12.2%,
tp8 `mc=256` +8.1%.

## Why, most likely: splitK tuned on an idle GPU

Among the shipped dense winners, splitK is selected almost exclusively at small M —
the decode regime:

| M bucket | n | winners with splitK>1 | median splitK |
|---|---|---|---|
| M≤8 | 39 | 87.2% | 6 |
| M 9–32 | 41 | 65.9% | 5 |
| M 33–256 | 142 | 28.9% | 4 |
| M>256 | 519 | 0.4% | 3 |

In the tuner's own timings those splitK kernels beat the best non-splitK candidate by
a median **21.6%** (M≤8) / **14.4%** (M 9–32). The tuner measures one GEMM at a time
on an otherwise idle GPU, where splitK is close to free: it recruits idle CUs and pays
only a reduction pass. Under production decode the GPU is not idle — attention, MoE
and all-reduce are co-resident — so those extra CUs are contended, and the reduction
pass, its extra global traffic, and the lost overlap are charged against TPOT.

This is consistent with all three concurrency regimes: at `mc=4` the GPU is closest to
the tuner's idle conditions and the tables do no harm; at `mc=16` decode dominates and
the splitK-heavy small-M rows regress hardest; at `mc=64/256` throughput is dominated
by large-M prefill GEMMs, where splitK is ~0% of winners and the tuned choices are
genuine wins.

The mechanism is reasoning over measured selections, not a measured in-situ kernel
comparison. A profile capture comparing per-kernel decode times between arms would
confirm or refute it.

Two things that could also matter and are *not* refuted:
- Within-libtype selection at small M is a near-tie — the winner beats the runner-up
  by a median **0.55%** (M≤8) / **0.67%** (M 9–32), far below any transferable margin.
  Selection there is essentially noise.
- Numerics changed and are unvalidated: `err_ratio` median 0.0124 at M≤8, max 0.0375
  across the table, consistent with splitK reductions.

Checked and **refuted**: that the tuner timed cache-resident weights. No shipped row
reports bandwidth above MI355X HBM peak (max 5.48 TB/s vs ~8 TB/s).

## Reproducing

Place the four CSVs in `aiter/configs/model_configs/`. aiter merges every table in
that directory at startup; confirm from the `merge tuned file under` log line that
both tuned filenames appear, and check for zero
`not found in a16w16 ... tune lookup table` errors. Compare against a run with the two
`*_tuned_*` files removed and the untuned ones left in place.

The full per-candidate profile output (1.69M rows, 431 MB across 10 shards) is not
committed; it is what the margin tables above were computed from.
