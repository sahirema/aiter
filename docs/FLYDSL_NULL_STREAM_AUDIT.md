# The flydsl NULL-stream default: mechanism, scope, and what is actually at risk

`stream: fx.Stream = fx.Stream(None)` appears as a launcher default in 40 places across 17
flydsl kernel modules. This note records what that default does, why it fails silently, which
sites were actually audited, and what the fix is. It is a report, not a code change.

## TL;DR

`Stream(None)` is the NULL/default stream. Under CUDA graph capture a launch on it is **not
recorded into the active graph**, so replay silently executes nothing. The hazard is not that
the default is wrong everywhere — on every path audited here a real stream is passed and the
default never binds. The hazard is that the parameter is *optional* at a boundary where the
failure mode is an empty graph rather than an exception, and that nothing enforces the caller's
side.

## 1. Mechanism

The authoritative statement is in the tree itself, at
`aiter/ops/flydsl/kernels/fused_compress_attn_hca.py:1197-1200`:

> `# CRITICAL: must pass current_stream when stream is None. Stream(None) =`
> `# NULL/default stream, which during CUDA graph capture produces an empty`
> `# graph entry (kernel launches don't get recorded into the active graph),`
> `# so replay is a no-op -> HCA boundaries silently never fire in decode CG.`

Followed immediately by the fix (`:1202-1204`):

```python
if stream is None:
    stream = torch.cuda.current_stream()
stream_obj = Stream(stream)
```

Three properties make this worth a written note rather than a lint rule:

1. **The failure is silent.** No exception, no wrong numbers at first — the kernel simply does
   not run on replay. Downstream state is stale rather than corrupt, which is harder to trace.
2. **It only manifests under capture.** Eager execution on the NULL stream is merely a
   serialization point, so eager tests pass and the defect appears only in a CUDA-graph decode
   path.
3. **The default is evaluated once.** `fx.Stream(None)` sits in a default argument of a launcher
   built inside an `@lru_cache`d factory, so it is constructed at first build for a given cache
   key and then shared by every later call under that key.

That comment's own back-reference, "Match v1 single-kernel pattern
(`fused_compress_attn.py:1381`)", has since drifted — the guards in that file are now at
`:2774` and `:2826`. The comment is right about the pattern and stale about the line.

## 2. Correction to the previously recorded premise

The prior internal record held that aiter PR **#4371** replaced a required `stream: fx.Stream`
positional with a NULL default and deleted a `torch.cuda.current_stream()` guard, and that the
`0029` patch carried the defect. **That is refuted on all three counts.**

- The `0029` patch (`0029-flydsl-fused-qk-norm-mrope-optest.patch`) is the *correct* exemplar,
  not the defect. It declares `stream: torch.cuda.Stream | None = None` (patch `:841`) and keeps
  a live guard at patch `:1079-1080`:
  `if stream is None: stream = torch.cuda.current_stream()`.
- The NULL default in `qk_norm_rope_quant.py` did not arrive via #4371.
  `git log -S 'fx.Stream(None)'` on that file returns **#3320** (the original feature commit) and
  **#4403** (a ruff pin), and nothing else.
- No guard was deleted. The pattern is an idiom that was present from the start.

The docstring previously cited at `qk_norm_rope_quant.py:967-970` is unrelated to streams and
should not be reintroduced as evidence.

## 3. Scope

40 occurrences of `Stream(None)` / `fx.Stream(None)` across 17 modules. Counting guards
**per kernel file** is misleading, because the guard usually belongs one layer up in the public
wrapper rather than in the kernels module:

| kernel module | guard in kernels file | guard in wrapper |
|---|---|---|
| `fused_compress_attn_hca.py` | 3 | — |
| `fused_compress_attn_hca_gfx1250.py` | 2 | — |
| `qk_norm_rope_quant.py` | 2 | — |
| `moe_sorting_kernel.py` | 2 | — |
| `gemm_a16w16_gfx950.py` | 1 | — |
| `dcp_topk_merge.py` | 0 | 2 (`aiter/ops/flydsl/dcp_topk_merge.py:161`) |
| `mla_reduce.py`, `moe_route_maps.py`, `moe_contiguous_psum.py`, `moe_g2l_lut.py`, `moe_gather_reduce.py`, `moe_fused_route_quant_scatter.py`, `flash_attn_func_gfx1201.py`, `flydsl_dispatch_combine_intranode_kernel.py`, `mega_moe_gfx1250/{combine,dispatch}.py` | 0 | not co-located; unaudited |

`dcp_topk_merge` is the reason the raw count cannot be trusted: its kernels file has no guard at
all, yet the wrapper passes `torch.cuda.current_stream(gathered_scores.device)` — and does so
device-bound, with a comment explaining that a bare `current_stream()` returns a stream on the
*ambient* device.

## 4. Verified vs unverified

**Audited, and safe on the current call path:**

- `qk_norm_rope_quant.py:2295` (the TDM launcher). Its caller passes a stream explicitly as the
  final positional at `:1672` (`stream if has_direct else Stream(stream)`), reached through the
  public entry `flydsl_qk_norm_rope_quant` (`:1394`) whose own guards sit at `:1634-1635` and
  `:1708-1709`. The default never binds.
- `dcp_topk_merge.py:476` — guarded in the wrapper, as above.
- `fused_compress_attn_hca.py` — guarded explicitly and deliberately, with the comment quoted
  in §1.

**Not audited:** the remaining ten modules in the last row of the table. Their wrappers are not
co-located, so establishing reachability means finding each public entry point individually. No
claim is made here that they are broken; equally, none is made that they are safe.

**Structural inconsistency worth noting regardless of reachability.** Inside
`qk_norm_rope_quant.py` the three launchers disagree with each other: `:779` and `:1359` declare
`stream: fx.Stream` as **required**, while `:2295` defaults it. A caller that omits the argument
gets a `TypeError` from two of them and a silent NULL-stream launch from the third. Whatever the
project decides the convention should be, one file should not hold both.

## 5. Recommended fix

Preferred, in order:

1. **Make the parameter required at the launcher boundary** (`stream: fx.Stream`, no default),
   matching `:779` and `:1359`. Turns a silent empty-graph bug into an immediate `TypeError` at
   the one place that can still be fixed cheaply. Cost: touches every call site, some external.
2. **Keep the parameter optional but resolve it in the wrapper**, i.e. the
   `if stream is None: stream = torch.cuda.current_stream()` pattern, device-bound as in
   `dcp_topk_merge.py:161`. Lower churn, preserves the ergonomic default, but leaves the trap in
   place for any future caller that bypasses the wrapper.

Option 1 is correct; option 2 is what the codebase already does where it does anything. A
`# noqa: B008` — present on nearly every one of these sites, in one case annotated "framework
idiom" — is what currently silences the linter that would otherwise flag the call-in-default.
Whichever option is chosen, that suppression should go with it.

## 6. Audit procedure for the remaining sites

For each `Stream(None)` default, the question is only ever *can a caller reach this default*:

1. Find the enclosing factory and what it returns.
2. Find every call site of the returned launcher, including external ones — the wrapper is not
   reliably at `aiter/ops/flydsl/<name>.py`.
3. Check whether the stream is passed at each. Note it is often passed **positionally as the
   final argument**, so grepping for `stream=` misses it.
4. If any path omits it, check whether that path can run under CUDA graph capture. If it can,
   it is a live silent-replay bug; if not, it is a latent trap.

Step 3 is where a grep-only audit goes wrong, and step 2 is where a same-directory assumption
does.
