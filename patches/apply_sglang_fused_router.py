#!/usr/bin/env python3
"""Wire the fused router+sort path into sglang. Default OFF.

Companion to apply_aiter_fused_router.py -- that one teaches aiter's
_fused_moe_impl to accept a `fused_router` payload instead of calling
moe_sorting; this one teaches sglang to produce that payload.

The seam already exists and is not invented here. `TopKConfig.output_format`
is checked FIRST in TopK.forward_cuda (topk.py:493), ahead of every backend
election, and `BypassedTopKOutput` already carries exactly what the fused
kernel needs -- router_logits plus the TopKConfig -- with a `.to_standard()`
that materialises routing the ordinary way. So the whole change is:

  1. declare the env flag,
  2. elect BYPASSED in TopK.__init__ when the flag is on and the STATIC
     conditions hold,
  3. in the aiter pre_permute, re-check the SHAPE-DEPENDENT conditions and
     either build the payload or call .to_standard() and carry on unchanged.

The static/dynamic split is the point. `output_format` is fixed per layer at
construction, but `can_fuse` depends on M, which changes every forward -- a
decode batch fuses and a 512-token prefill must not. Electing statically and
falling back dynamically is what lets one layer do both.

Every fallback in this patch returns None rather than raising, so an
unsupported shape, a missing fused_router_sort module, or an UNPATCHED aiter
wheel all degrade to the ordinary router + moe_sorting pair. The two patches
are applied independently, so "sglang patched, aiter not" is a real state and
must not be a crash.

Exact-match, not a unified diff: line numbers fail opaquely against a drifted
tree, whereas a missing anchor here aborts loudly and writes nothing.

Usage:
    python apply_sglang_fused_router.py <sglang/python/sglang root> [--write]
"""
import argparse
import ast
import hashlib
import pathlib
import sys

FLAG = "SGLANG_FUSE_MOE_ROUTER_SORT"

# ---------------------------------------------------------------- environ.py
ENV_OLD = "    SGLANG_STRICT_CONFIG_MUTATION = EnvBool(False)\n"
ENV_NEW = (
    f"    # Fuse the MoE router, moe_sorting and the moe_buf zero-init into a\n"
    f"    # single Triton kernel on the aiter path. Requires an aiter patched\n"
    f"    # with fused_moe(..., fused_router=...); falls back silently if not.\n"
    f"    {FLAG} = EnvBool(False)\n"
    + ENV_OLD
)

# ------------------------------------------------------------------- topk.py
TOPK_OLD = "        self.is_fp4_experts = is_fp4_experts\n"
TOPK_NEW = (
    "        self.is_fp4_experts = is_fp4_experts\n"
    "        # Router+sort fusion (aiter path): defer routing so the patched\n"
    "        # aiter _fused_moe_impl can run router + moe_sorting + moe_buf\n"
    "        # zero-init as one kernel. Deliberately placed AFTER the waterfill\n"
    "        # block above, which forces STANDARD -- waterfill must win.\n"
    "        # Only the layer-constant conditions are tested here. The ones that\n"
    "        # depend on the batch (M) or on EP state are re-tested every forward\n"
    "        # in pre_permute_standard_to_aiter, which falls back through\n"
    "        # BypassedTopKOutput.to_standard(). A bias is required because the\n"
    "        # fused kernel always reads one; models without routing bias keep\n"
    "        # the unfused path.\n"
    "        if (\n"
    "            output_format is None\n"
    f"            and envs.{FLAG}.get()\n"
    "            and not use_grouped_topk\n"
    "            and num_fused_shared_experts == 0\n"
    "            and custom_routing_function is None\n"
    "            and correction_bias is not None\n"
    "            and get_moe_runner_backend().is_aiter()\n"
    "        ):\n"
    "            output_format = TopKOutputFormat.BYPASSED\n"
)

# ------------------------------------------------------ moe_runner/aiter.py
IN_OLD = """    num_local_tokens: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER
"""
IN_NEW = """    num_local_tokens: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None
    # (router_logits, bias, topk, gate_kwargs) for the fused router+sort path.
    # When set, routing has NOT happened yet -- aiter does it inside
    # _fused_moe_impl, and topk_ids/topk_weights below are empty placeholders
    # that exist only so the zero-token early-out can read topk from shape[-1].
    fused_router: Optional[tuple] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER
"""

RUN_OLD = """        if self.config.no_combine:
            extra["no_combine"] = True

        output = fused_moe(
"""
RUN_NEW = """        if runner_input.fused_router is not None:
            extra["fused_router"] = runner_input.fused_router
        if self.config.no_combine:
            extra["no_combine"] = True

        output = fused_moe(
"""

HELPER_OLD = '''@register_pre_permute("standard", "aiter")
def pre_permute_standard_to_aiter(
'''
HELPER_NEW = '''@functools.cache
def _aiter_fused_moe_supports_fused_router() -> bool:
    """Probe whether the installed aiter.fused_moe accepts `fused_router`.

    The sglang and aiter patches are applied independently, so an sglang that
    elects BYPASSED against a stock aiter wheel is a reachable state. Detect it
    once and fall back, rather than raising TypeError on every forward.
    """
    from aiter.fused_moe import fused_moe

    return "fused_router" in inspect.signature(fused_moe).parameters


def _try_build_fused_router(
    topk_output,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> Optional[tuple]:
    """Build the fused router+sort payload, or None to use the unfused path.

    Returns None rather than raising for every unsupported case: this runs on
    the hot path of every forward, and an unsupported shape must cost a
    fallback, not a crash. The checks here are the ones TopK.__init__ could not
    make because they depend on the batch or on runtime EP state.
    """
    if not _aiter_fused_moe_supports_fused_router():
        return None
    if runner_config.apply_router_weight_on_input:
        # Would need topk_weights materialised to pre-scale hidden_states.
        return None
    if quant_info.expert_mask is not None:
        return None
    if topk_output.num_token_non_padded is not None:
        return None
    if topk_output.expert_location_dispatch_info is not None:
        return None

    cfg = topk_output.topk_config
    if cfg.correction_bias is None:
        return None

    logits = topk_output.router_logits
    # aiter derives its expert count from w1.shape[0]; the fused kernel asserts
    # scores.shape[1] == bias.shape[0] == that value. Redundant-expert setups
    # make them diverge, which would be an assert rather than a fallback -- so
    # check it here, where falling back is still possible.
    if logits.shape[1] != quant_info.w13_weight.shape[0]:
        return None
    if cfg.correction_bias.shape[0] != logits.shape[1]:
        return None

    try:
        from fused_router_sort import can_fuse
    except ImportError:
        return None

    ok, _why = can_fuse(
        logits.shape[0],
        logits.shape[1],
        cfg.top_k,
        num_expert_group=cfg.num_expert_group or 1,
        topk_group=cfg.topk_group or 1,
        num_fused_shared_experts=cfg.num_fused_shared_experts,
        expert_mask=quant_info.expert_mask,
    )
    if not ok:
        return None

    return (
        logits,
        cfg.correction_bias,
        cfg.top_k,
        dict(
            scoring_func=cfg.scoring_func,
            renormalize=cfg.renormalize,
            routed_scaling_factor=float(cfg.routed_scaling_factor or 1.0),
            apply_routed_scaling_factor_on_output=bool(
                cfg.apply_routed_scaling_factor_on_output
            ),
            # Consumed by can_fuse on the aiter side and popped before the
            # payload reaches fused_router_sort, which has no such parameters.
            num_expert_group=cfg.num_expert_group or 1,
            topk_group=cfg.topk_group or 1,
            num_fused_shared_experts=cfg.num_fused_shared_experts,
        ),
    )


@register_pre_permute("standard", "aiter")
def pre_permute_standard_to_aiter(
'''

PRE_OLD = """    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(torch.float32)
"""
PRE_NEW = """    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    if TopKOutputChecker.format_is_bypassed(topk_output):
        fused_router = _try_build_fused_router(topk_output, quant_info, runner_config)
        if fused_router is not None:
            # Routing is deferred into aiter. The placeholders are empty on the
            # token axis but keep the topk width, which is all run() reads them
            # for on the zero-token early-out.
            top_k = topk_output.topk_config.top_k
            return AiterRunnerInput(
                hidden_states=hidden_states,
                topk_ids=hidden_states.new_empty((0, top_k), dtype=torch.int32),
                topk_weights=hidden_states.new_empty((0, top_k), dtype=torch.float32),
                quant_type=quant_info.quant_type,
                fused_router=fused_router,
            )
        # Not fusable at this shape -- materialise routing and continue exactly
        # as the unpatched path does.
        topk_output = topk_output.to_standard()

    topk_weights, topk_ids, _ = topk_output
    topk_weights = topk_weights.to(torch.float32)
"""

EDITS = [
    ("srt/environ.py", "1/5 env flag declaration", ENV_OLD, ENV_NEW, 1),
    ("srt/layers/moe/topk.py", "2/5 BYPASSED election in TopK.__init__",
     TOPK_OLD, TOPK_NEW, 1),
    ("srt/layers/moe/moe_runner/aiter.py", "3/5 AiterRunnerInput.fused_router",
     IN_OLD, IN_NEW, 1),
    ("srt/layers/moe/moe_runner/aiter.py", "4/5 forward fused_router to aiter",
     RUN_OLD, RUN_NEW, 1),
    ("srt/layers/moe/moe_runner/aiter.py", "5a/5 payload builder + probe",
     HELPER_OLD, HELPER_NEW, 1),
    ("srt/layers/moe/moe_runner/aiter.py", "5b/5 bypassed handling in pre_permute",
     PRE_OLD, PRE_NEW, 1),
]

# Idempotency is checked per FILE with a marker, not per edit by "is the
# replacement already there". Several replacements deliberately KEEP their own
# anchor (ENV_NEW ends with ENV_OLD), so an anchor-based check silently permits
# a second application and duplicates the inserted block.
MARKERS = {
    "srt/environ.py": FLAG,
    "srt/layers/moe/topk.py": FLAG,
    "srt/layers/moe/moe_runner/aiter.py": "_try_build_fused_router",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="path to sglang/python/sglang")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    root = pathlib.Path(args.root)

    # Fail before touching anything if any target is already patched.
    for rel, marker in MARKERS.items():
        path = root / rel
        if not path.is_file():
            print(f"ABORT: missing {path}")
            return 1
        if marker in path.read_text():
            print(f"ABORT: {rel} already contains {marker!r} -- already patched")
            return 1

    bodies: dict[pathlib.Path, str] = {}
    for rel, name, old, new, want in EDITS:
        path = root / rel
        if path not in bodies:
            bodies[path] = path.read_text()
        body = bodies[path]
        got = body.count(old)
        if got != want:
            print(f"ABORT: {name} in {rel}: expected {want} occurrence(s), found {got}")
            return 1
        bodies[path] = body.replace(old, new, want)
        print(f"  ok  {name}  ({want} site)")

    for path, body in bodies.items():
        try:
            ast.parse(body)
        except SyntaxError as e:
            print(f"ABORT: patched {path.name} does not parse: {e}")
            return 1
    print("all patched files parse")

    if not args.write:
        print("\nDRY RUN -- nothing written. Re-run with --write.")
        return 0
    for path, body in bodies.items():
        path.write_text(body)
        print(f"WROTE {path}  sha256 {hashlib.sha256(body.encode()).hexdigest()}")
    print(f"\nEnable at runtime with {FLAG}=1 (default off).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
