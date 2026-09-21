#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Thread an optional `fused_router` bundle into aiter's fused_moe, so the
router+sort fusion can replace the `moe_sorting` call without reimplementing
anything else in _fused_moe_impl.

WHY AN APPLIER AND NOT A .patch
    A unified diff carries line numbers and fails opaquely against a tree that
    has drifted. This matches on exact source text, asserts the expected number
    of occurrences at every step, records the pre-edit sha256, and refuses to
    write anything if any anchor is wrong -- so a drifted aiter produces a named
    error instead of a half-applied file. Satisfies uplift protocol item 8(a):
    the recorded base sha is the build-time provenance guard.

WHY SIX EDITS
    fused_moe -> fused_moe_ -> _fused_moe_impl forward every argument by
    explicit keyword; neither layer takes **kwargs (verified at fused_moe.py:578
    and :701). A new parameter declared only on the outer signature would be
    accepted and then silently dropped at the first forwarding call -- so all
    three signatures AND both forwarding call sites must change, plus the one
    use site. Five of six edits are pure plumbing; only the last changes
    behaviour.

USAGE
    python apply_aiter_fused_router.py /path/to/aiter/fused_moe.py [--write]
    Dry by default: prints the edits and the resulting diff, writes nothing.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import pathlib
import sys

# sha256 of the file this was authored against: the shipped-image aiter
# extracted to uplift_work/container_src (md5 76504f9954aa0f69c11ffbac5f22d167).
# A mismatch is not fatal -- the anchors are matched on text -- but it is
# reported, because "applied cleanly to a different aiter" is a claim that
# needs to be visible rather than assumed.
BASE_SHA256 = "f553937c94bc0864cc1bed747dc446c6ed7ff2ceaf9205ce4f4bff39916d202f"

EDITS: list[tuple[str, int, str, str]] = [
    # (label, expected occurrences, old, new)
    (
        "1/5 fused_moe signature",
        1,
        "    shared_expert_id: int = -1,\n"
        "    out: torch.Tensor | None = None,\n"
        "    residual: torch.Tensor | None = None,\n"
        "):\n",
        "    shared_expert_id: int = -1,\n"
        "    out: torch.Tensor | None = None,\n"
        "    residual: torch.Tensor | None = None,\n"
        "    fused_router: tuple | None = None,\n"
        "):\n",
    ),
    (
        "2/5 _fused_moe_impl signature",
        1,
        "    _stage2_extra_args: dict | None = None,\n"
        "    out: torch.Tensor | None = None,\n"
        "    residual: torch.Tensor | None = None,\n"
        ") -> torch.Tensor:\n",
        "    _stage2_extra_args: dict | None = None,\n"
        "    out: torch.Tensor | None = None,\n"
        "    residual: torch.Tensor | None = None,\n"
        "    # (scores, bias, topk, gate_kwargs) -- when set, the router and the\n"
        "    # sort run as one kernel and topk_weight/topk_ids are unused.\n"
        "    fused_router: tuple | None = None,\n"
        ") -> torch.Tensor:\n",
    ),
    (
        # A tuple has no torch.library schema type (it is absent from
        # torch._library.infer_schema.SUPPORTED_PARAM_TYPES), so `fused_router`
        # cannot cross the @torch_compile_guard-registered `fused_moe_` --
        # infer_schema raises AttributeError on types.UnionType while reaching
        # for the Tuple diagnostic. `fused_moe_` is a PURE forwarder to
        # `_fused_moe_impl` (its whole body is that one call), so the fused path
        # dispatches straight to the impl and skips no logic. The unfused path
        # is untouched and remains a registered custom op -- which matters for
        # A/B: the control must not change.
        "3/5 fused path bypasses the registered wrapper",
        1,
        "    result = fused_moe_(\n",
        "    _impl = _fused_moe_impl if fused_router is not None else fused_moe_\n"
        "    # `fused_moe_` is a REGISTERED custom op: its schema has 27 params and\n"
        "    # none of them is `fused_router`, so passing the kwarg even as None is a\n"
        "    # hard dispatch error (\'expected at most 27 argument(s) but received\n"
        "    # 28\'), which is raised at CUDA-graph-capture time on the unfused\n"
        "    # fallback. The kwarg therefore travels only on the fused path, where\n"
        "    # `_impl` is the plain-Python `_fused_moe_impl`.\n"
        "    _fused_kw = {\'fused_router\': fused_router} if fused_router is not None else {}\n"
        "    result = _impl(\n",
    ),
    (
        # Anchored on the trailing `# plumbing:` comment: the two call-site tails
        # are byte-identical, and only this one is followed by that comment. The
        # other tail (fused_moe_ -> _fused_moe_impl) must NOT be edited, since
        # fused_router is no longer a parameter of fused_moe_.
        "4/5 forward fused_router at the fused_moe call site only",
        1,
        "        gate_mode=gate_mode,\n"
        "        out=out,\n"
        "        residual=residual,\n"
        "    )\n"
        "\n"
        "    # plumbing: optional caller-provided output buffer",
        "        gate_mode=gate_mode,\n"
        "        out=out,\n"
        "        residual=residual,\n"
        "        **_fused_kw,\n"
        "    )\n"
        "\n"
        "    # plumbing: optional caller-provided output buffer",
    ),
    (
        "5/5 the sort call site (the only behavioural change)",
        1,
        "    else:\n"
        "        sorting_ret = moe_sorting(\n"
        "            topk_ids,\n"
        "            topk_weight,\n"
        "            global_E,\n"
        "            model_dim,\n"
        "            dtype,\n"
        "            block_size_M,\n"
        "            expert_mask,\n"
        "            num_local_tokens,\n"
        "            moe_sorting_dispatch_policy,\n",
        "    elif fused_router is not None:\n"
        "        # Router + sort + moe_buf zero-init as a single kernel. Everything\n"
        "        # downstream of here is untouched: the 5-tuple below has the same\n"
        "        # contract moe_sorting returns, so fused_moe_2stages cannot tell\n"
        "        # the difference. Guarded by can_fuse() on the caller's side and\n"
        "        # again here, because the unsupported cases (grouped routing, EP,\n"
        "        # fused shared experts) would sort silently wrong rather than fail.\n"
        "        from fused_router_sort import can_fuse, fused_router_sort\n"
        "\n"
        "        _scores, _bias, _topk, _gate_kw = fused_router\n"
        "        # Routing-topology keys are consumed by can_fuse and must NOT reach\n"
        "        # fused_router_sort, whose signature has no such parameters -- a bare\n"
        "        # **splat of the caller's dict would raise TypeError. Popping here (not\n"
        "        # omitting them caller-side) keeps the can_fuse check non-vacuous.\n"
        "        _gate_kw = dict(_gate_kw)\n"
        "        _n_group = int(_gate_kw.pop('num_expert_group', None) or 1)\n"
        "        _topk_group = int(_gate_kw.pop('topk_group', None) or 1)\n"
        "        _n_shared = int(_gate_kw.pop('num_fused_shared_experts', None) or 0)\n"
        "        _ok, _why = can_fuse(\n"
        "            _scores.shape[0], global_E, _topk,\n"
        "            num_expert_group=_n_group, topk_group=_topk_group,\n"
        "            num_fused_shared_experts=_n_shared,\n"
        "            expert_mask=expert_mask, num_local_tokens=num_local_tokens,\n"
        "        )\n"
        "        assert _ok, f'fused_router requested but unsupported here: {_why}'\n"
        "        # Load-bearing for TWO reasons -- do not delete as redundant.\n"
        "        # (1) the flat sort layout is genuinely not implemented here; and\n"
        "        # (2) it is what makes `out_buf` below correct. aiter's own general\n"
        "        #     branch uses `out if (accumulate and not metadata.flat) else None`\n"
        "        #     (fused_moe.py:998-1002); ours uses `out if _accum else None`.\n"
        "        #     Those are equivalent ONLY while metadata.flat is False. Removing\n"
        "        #     this assert silently passes `out` as the moe_buf on the flat path.\n"
        "        assert not metadata.flat, 'fused_router does not implement the flat sort layout'\n"
        "        assert not need_local_topk_ids, (\n"
        "            'fused_router does not emit local expert ids; can_fuse rejects '\n"
        "            'expert_mask, which is what would set this'\n"
        "        )\n"
        "        _accum = not stage2_uses_route_reduce(metadata.stage2)\n"
        "        # Unpacked here rather than through `sorting_ret`: the else branch\n"
        "        # below unpacks inside itself, so a branch that only sets sorting_ret\n"
        "        # leaves sorted_ids undefined at check_route_bucket_metadata.\n"
        "        (\n"
        "            sorted_ids,\n"
        "            sorted_weights,\n"
        "            sorted_expert_ids,\n"
        "            num_valid_ids,\n"
        "            moe_buf,\n"
        "        ) = fused_router_sort(\n"
        "            _scores, _bias, _topk, global_E, model_dim, dtype,\n"
        "            block_size=block_size_M, accumulate=_accum,\n"
        "            out_buf=(out if _accum else None),\n"
        "            **_gate_kw,\n"
        "        )\n"
        "        local_topk_ids = None\n"
        "    else:\n"
        "        sorting_ret = moe_sorting(\n"
        "            topk_ids,\n"
        "            topk_weight,\n"
        "            global_E,\n"
        "            model_dim,\n"
        "            dtype,\n"
        "            block_size_M,\n"
        "            expert_mask,\n"
        "            num_local_tokens,\n"
        "            moe_sorting_dispatch_policy,\n",
    ),
]


MARKER = "elif fused_router is not None:"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="path to aiter/fused_moe.py")
    ap.add_argument("--write", action="store_true", help="actually modify the file")
    args = ap.parse_args()

    p = pathlib.Path(args.target)
    src = p.read_text()
    sha = hashlib.sha256(src.encode()).hexdigest()
    print(f"target   {p}")
    print(f"base sha256 {sha}")
    if BASE_SHA256 not in ("", sha) and len(BASE_SHA256) == 64:
        print(f"  NOTE: differs from the authored-against base {BASE_SHA256}")

    # Distinguish "already patched" from "tree drifted". Both surface as a
    # missing anchor (edit 1 consumes its own anchor, so a second run finds 0),
    # but they need opposite responses: re-derive the anchor vs. do nothing.
    if MARKER in src:
        print(f"\nABORT: target already contains {MARKER!r} -- already patched, "
              "nothing to do.")
        return 1

    out = src
    for label, n, old, new in EDITS:
        found = out.count(old)
        if found != n:
            print(f"\nABORT at {label}: expected {n} occurrence(s), found {found}.")
            print("  aiter has drifted from the tree this was written against; "
                  "re-derive the anchor rather than forcing it.")
            return 1
        out = out.replace(old, new)
        print(f"  ok  {label}  ({n} site{'s' if n > 1 else ''})")

    if out == src:
        print("\nno change produced -- already applied?")
        return 1
    diff = difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                                "a/fused_moe.py", "b/fused_moe.py", n=2)
    body = "".join(diff)
    print(f"\n--- resulting diff ({body.count(chr(10))} lines) ---")
    print(body)
    if not args.write:
        print("DRY RUN -- nothing written. Re-run with --write.")
        return 0
    p.write_text(out)
    print(f"WROTE {p}  new sha256 {hashlib.sha256(out.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
