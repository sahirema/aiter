# MiniMax-M3-MXFP4 dense MXFP4 GEMM tuning (gfx950 / MI355X)

Harness + shape list for tuning the `GEMM-AFP4WFP4` Triton family for
MiniMax-M3-MXFP4. **No measured rows are committed here.** The JSONs are
produced only by running the script on a GPU node.

## Which path this targets

sglang's quark W4A4 MXFP4 linear has **no ASM/CK branch**. `apply_weights`
(`quark_w4a4_mxfp4.py:299-341`) dispatches to one of three Triton aiter entries,
and the general case is line 341 -> `gemm_afp4wfp4` (`shuffle=False`), so the
family consulted is `GEMM-AFP4WFP4` — *not* `GEMM-AFP4WFP4_PRESHUFFLED`.

Lookup chain:
`gemm_afp4wfp4` -> `_get_config` (`_triton_kernels/gemm/basic/gemm_afp4wfp4.py:718-737`)
-> `get_gemm_config` (`utils/gemm_config_utils.py:113`)
-> `resolve_config_dir` (`utils/config_utils.py:77`)
-> `configs/<arch>/<backend>/gemm/<d_type>/`.

**Key on a lookup:** the specialized file is keyed on `(N, K)` only
(`gemm_config_utils.py:74-86`); `M` then selects a bucket *inside* the chosen
file via `M_LEQ_<bound>` / `M_GEQ_<bound>` / `any`
(`gemm_config_utils.py:89-100`). dtype, split-k and gate mode are not part of
the key.

**Behaviour on a miss:** silent. `load_config_json(..., required=False)` returns
`None` (`config_utils.py:45-53`), the loop leaves `config_dict` as `DEFAULT.json`
and `is_tuned = False`. Nothing raises; the kernel just runs an untuned default.
A `KeyError` is raised only if no `M` bucket matches at all
(`gemm_config_utils.py:102-105`).

## Shapes

See `shapes_m3_mxfp4.csv`. Derivation summary:

Among the dense linears, **only** `shared_experts.{gate,up,down}_proj` are MXFP4.
Every `q/k/v/o_proj` (all 60 layers), the 3 dense-MLP layers, `index_q/k_proj`,
`lm_head` and the whole vision tower are listed in `quantization_config.exclude`;
confirmed against the checkpoint, where the only modules carrying
`.weight_scale` are the MoE experts `w1/w2/w3` and the shared-expert projections
(`model.safetensors.index.json`). The MoE experts are MXFP4 but run on the
fused-MoE path, not `gemm_afp4wfp4`.

With `hidden_size=6144`, `intermediate_size=3072`, `n_shared_experts=1`
(`minimax_m3.py:392-404`, `270-289` on `upstream/release/v0.5.20`):
`gate_up_proj` is `MergedColumnParallelLinear(hidden, [I]*2)`, so local
`N = 2*I/tp`; `down_proj` is `RowParallelLinear(I, hidden)`, so local
`N = hidden`, `K = I/tp`.

| variant | module | N | K |
|---|---|---|---|
| tp_sharded, tp=2 | gate_up_proj | 3072 | 6144 |
| tp_sharded, tp=2 | down_proj | 6144 | 1536 |
| tp_sharded, tp=4 | gate_up_proj | 1536 | 6144 |
| tp_sharded, tp=4 | down_proj | 6144 | 768 |
| tp_sharded, tp=8 | gate_up_proj | 768 | 6144 |
| tp_sharded, tp=8 | down_proj | 6144 | 384 |
| replicated (DeepEP) | gate_up_proj | 6144 | 6144 |
| replicated (DeepEP) | down_proj | 6144 | 3072 |

Eight distinct `(N,K)` keys, no collisions. All three TP degrees that fit an
8-GPU MI355X node are listed: the canonical MiniMax-M3 config is tp2, and
protocol v4 §2 derives TP variants from it, so tp4/tp8 arms would otherwise
run untuned. The replicated pair appears once because `minimax_m3.py:392-404`
sets `tp_size=1` when `get_moe_a2a_backend().is_deepep()` — that variant is
TP-independent. Sweeping all eight costs tuning time only; the lookup is keyed
on `(N,K)`, so a JSON for a TP degree you do not run is never loaded.

**Reachability.** `shared_experts` is built only when
`num_fused_shared_experts == 0`. On `upstream/release/v0.5.20` — the branch the
benchmark image is cut from — the ROCm path hits
`if not _is_cuda: return "Shared experts fusion currently requires CUDA devices."`
(`minimax_m3_vl.py:155-156`, `minimax_m3.py:1615-1616`), so fusion is **off** and the
module **is** built. Commit `241a5b9823` (2026-09-16, *"allow shared-experts
fusion on ROCm gfx942 and newer"*) is **not** an ancestor of that branch but is
on upstream main; after it, fusion is on for gfx950 and **none** of these shapes
run. Re-check this gate against whatever image you tune for.

`K` is the **logical** (unpacked) K: `test_gemm_afp4wfp4.py:52` builds `x` as
`(M, K//2)`, and `_get_config` doubles the packed K back before forming the
filename (`gemm_afp4wfp4.py:725-726`).

## Running it (GPU node required)

    script -qec "srun --jobid=<JOBID> --overlap bash -lc '$HOME/aiter/tuning/m3_gfx950_mxfp4/tune_m3_gfx950_mxfp4.sh'" /dev/null

`rc=0` from `srun` is not evidence anything ran. Check the log at
`tuning/m3_gfx950_mxfp4/out/tune_m3_gfx950_mxfp4.log` for the sentinel
`=== M3 TUNE COMPLETE ===` and one `screen done` line per `(M,N,K)` case.

## Installing the generated configs (separate, reviewed step)

The script deliberately does **not** write into the config tree. Copy with the
arch prefix dropped — the nested layout keys arch by directory:

    cp out/GEMM-AFP4WFP4-N=3072-K=6144.json \
       $HOME/aiter/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_afp4wfp4/

Two gotchas from `tunning/README.md`: the family's `DEFAULT.json` must already be
in place, and config reads are cached per path **including negative results**
(`config_utils.py:36-38`), so restart the process after copying.

> Note: `view-screen.py:70` guesses an **arch-prefixed** name
> (`gfx950-GEMM-AFP4WFP4-...`), which is the old flat convention and would never
> be loaded by the current nested layout. The script therefore passes
> `--json-prefix GEMM-AFP4WFP4` explicitly.

## Verifying a shape is actually tuned

`gemm_tune_check` reports `is_tuned` straight off the real lookup. Note it takes
the **packed** K (logical/2):

    from aiter.ops.triton.utils._triton.gemm_tune_check import gemm_tune_check
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    gemm_tune_check(gemm_afp4wfp4, N=3072, K=6144 // 2, M=16, shuffle=False)

Run it before and after installing: it must flip `False` -> `True`.

## Prior art

`upstream/users/msaffari-amd/aiter/tune-mxfp4-proj-upproj-gemm-gfx950`
(commit `7bc5dbc38`) adds four MiniMax-M3 MXFP4 JSONs. They are **not** drop-in:

* They use the old flat `configs/gemm/gfx950-*.json` layout — the nested tree
  does not exist at that commit — so dropping them into this tree unchanged
  leaves them unread.
* Its two plain-family files are the **tp4** shapes of the same shared-expert
  MLP: `N=1536-K=6144` is `gate_up` at `2*3072/4`, `N=6144-K=768` is `down` at
  `3072/4`. Those two are now in scope here (tp4 rows above), so they are a
  useful **cross-check** on the tp4 results — but they still need relocating out
  of the flat layout, and they were tuned at a different aiter commit.
* Its two `_PRESHUFFLED` files are on a family this path never consults
  (line 341 calls the plain entry). One of them, `N=6144-K=1536`, does coincide
  with the tp2 `down_proj` shape derived above, but it was tuned for the
  preshuffled kernel and its parameters are not transferable.

So the values are measured, but the shape/family mapping differs. Do not copy
them without re-deriving the mapping.
