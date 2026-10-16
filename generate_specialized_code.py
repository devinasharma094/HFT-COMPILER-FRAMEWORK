"""
generate_specialized_code.py  (v2: exact interval-based elimination)

Phase 2 compiler pass. Reads model.json (XGBoost) and vec.json (the VEC) and emits C:

  baseline_model.c       full unmodified model (fallback + benchmark baseline)
  specialized_<regime>.c per-regime kernels
  guards.c               interval guards (exact doubles, same box used by the compiler)
  dispatch.c             guard -> specialized kernel, else baseline

TIER 1 (exact): inside regime R's guard box B (a closed box over the 3 features), a split
    "x < t"  is always true   if  hi(x) < t      (or the tracked upper bound is strict and <= t)
    "x < t"  is always false  if  lo(x) >= t
The compiler removes the comparison and emits only the surviving child. The box is narrowed as
we descend (yes: x < t, no: x >= t), so deeper splits can be removed too. Because this is pure
interval reasoning over the SAME box the guard checks, the specialized kernel returns exactly
the baseline's output whenever the guard passes. Each elimination cites the VEC feature-interval
clause that licenses it.

TIER 2 (probabilistic, can never change an output): where a branch clause's Clopper-Pearson upper
bound on firing "yes" is small, the retained comparison is wrapped in __builtin_expect (layout hint).

Usage:
    python generate_specialized_code.py --model model.json --vec vec.json --out_dir generated_c
"""

import argparse
import json
import math
import os

import numpy as np
from xgboost import XGBClassifier
from vec_ir import VECIR

FEATURE_COLS = ["realized_vol", "trade_intensity", "avg_trade_size"]
REGIMES = ["calm", "normal", "volatile"]

HINT_MARGIN = 0.10  # Tier 2: if upper bound on P(yes) <= this, mark the 'yes' side UNLIKELY


def f32(x) -> float:
    """Round to float32 (XGBoost's native precision) and return as exact Python float."""
    return float(np.float32(x))


def flit(x) -> str:
    """C float literal that round-trips the float32 value exactly (e.g. 44f -> 44.0f)."""
    s = str(np.float32(x))
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s + "f"


def load_tree_structure(model: XGBClassifier) -> list:
    dump = model.get_booster().get_dump(dump_format="json")
    return [json.loads(t) for t in dump]


def get_base_margin(model: XGBClassifier) -> float:
    """logit(base_score) so the C model matches XGBoost's margin; falls back to 0.0 (base_score 0.5)."""
    try:
        cfg = json.loads(model.get_booster().save_config())
        raw = str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[] ")
        p = float(raw)
        return math.log(p / (1.0 - p))
    except Exception as e:  # noqa
        print(f"  WARNING: could not read base_score ({e}); assuming margin 0.0")
        return 0.0


def index_branch_clauses(vec: dict) -> dict:
    return {(c["regime"], c["tree_index"], c["node_id"]): c for c in vec["branch_clauses"]}


def regime_box(vec: dict, regime: str) -> dict:
    """feature -> [lo, hi, hi_is_strict]; exactly the values the guard is emitted with."""
    return {
        feat: [float(iv["lower_bound"]), float(iv["upper_bound"]), False]
        for feat, iv in vec["feature_intervals"][regime].items()
    }


def emit_node_c(node, tree_index, regime, branch_index, vec_ir, box, lines, indent, stats):
    if "leaf" in node:
        lines.append(f"{indent}return {f32(node['leaf'])!r};")
        return

    feat = node["split"]
    if feat not in FEATURE_COLS and feat[:1] == "f" and feat[1:].isdigit():
        feat = FEATURE_COLS[int(feat[1:])]

    t = f32(node["split_condition"])
    nid = f"{tree_index}-{node['nodeid']}"
    clause_id = (
        f"{regime}-{tree_index}-{nid.split(chr(45), 1)[1]}"
        if regime
        else None
    )

    if clause_id:
        lines.append(f"/* VEC_REGION_BEGIN {clause_id} */")

    yes_child = next(
        c for c in node["children"] if c["nodeid"] == node["yes"]
    )
    no_child = next(
        c for c in node["children"] if c["nodeid"] == node["no"]
    )

    if box is not None:
        lo, hi, strict = box[feat]
        cite = (
            f"VEC feature_interval {regime}.{feat} = "
            f"[{lo!r}, {hi!r}]"
        )

        if hi < t or (strict and hi <= t):
            lines.append(
                f"{indent}/* ELIMINATED (exact, always yes): "
                f"{feat} < {t!r} (node {nid}); "
                f"licensed by VEC-IR clause {clause_id}: {cite} */"
            )
            stats["eliminated"] += 1

            emit_node_c(
                yes_child,
                tree_index,
                regime,
                branch_index,
                vec_ir,
                box,
                lines,
                indent,
                stats,
            )

            if clause_id:
                lines.append(
                    f"/* VEC_REGION_END {clause_id} */"
                )
            return

        if lo >= t:
            lines.append(
                f"{indent}/* ELIMINATED (exact, always no): "
                f"{feat} < {t!r} (node {nid}); "
                f"licensed by VEC-IR clause {clause_id}: {cite} */"
            )
            stats["eliminated"] += 1

            emit_node_c(
                no_child,
                tree_index,
                regime,
                branch_index,
                vec_ir,
                box,
                lines,
                indent,
                stats,
            )

            if clause_id:
                lines.append(
                    f"/* VEC_REGION_END {clause_id} */"
                )
            return

    # Split survives.
    # Tier 2 layout hint from the certified firing-rate bound.
    cond = f"{feat} < {flit(t)}"

    clause = (
        vec_ir.find_clause(
            f"{regime}-{tree_index}-{nid.split(chr(45), 1)[1]}"
        )
        if regime
        else None
    )

    if clause is not None and clause.branch_upper_bound <= HINT_MARGIN:
        cond = f"UNLIKELY({cond})"
        stats["hinted"] += 1

    stats["kept"] += 1

    if box is not None:
        yes_box = {k: list(v) for k, v in box.items()}
        yes_box[feat][1], yes_box[feat][2] = t, True

        no_box = {k: list(v) for k, v in box.items()}
        no_box[feat][0] = max(no_box[feat][0], t)
    else:
        yes_box = no_box = None

    lines.append(
        f"{indent}if ({cond}) {{ /* node {nid} */"
    )

    emit_node_c(
        yes_child,
        tree_index,
        regime,
        branch_index,
        vec_ir,
        yes_box,
        lines,
        indent + "    ",
        stats,
    )

    lines.append(f"{indent}}} else {{")

    emit_node_c(
        no_child,
        tree_index,
        regime,
        branch_index,
        vec_ir,
        no_box,
        lines,
        indent + "    ",
        stats,
    )

    lines.append(f"{indent}}}")

    if clause_id:
        lines.append(
            f"/* VEC_REGION_END {clause_id} */"
        )


def generate_model_file(trees, regime, branch_index, vec, vec_ir, base_margin, out_path) -> dict:
    stats = {"eliminated": 0, "kept": 0, "hinted": 0}
    suffix = f"_{regime}" if regime else "_baseline"
    box = regime_box(vec, regime) if regime else None
    lines = [
        "/* Auto-generated by generate_specialized_code.py */",
        "#include <math.h>",
        "#ifndef UNLIKELY",
        "#define UNLIKELY(x) __builtin_expect(!!(x), 0)",
        "#endif",
        "",
    ]
    for i, tree in enumerate(trees):
        lines.append(f"static double tree_{i}{suffix}(double realized_vol, double trade_intensity, double avg_trade_size) {{")
        emit_node_c(tree, i, regime, branch_index, vec_ir, box, lines, "    ", stats)
        lines.append("}")
        lines.append("")
    lines.append(f"double predict{suffix}(double realized_vol, double trade_intensity, double avg_trade_size) {{")
    lines.append(f"    double sum = {base_margin!r};")
    for i in range(len(trees)):
        lines.append(f"    sum += tree_{i}{suffix}(realized_vol, trade_intensity, avg_trade_size);")
    lines.append("    return 1.0 / (1.0 + exp(-sum));  /* sigmoid, binary:logistic */")
    lines.append("}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return stats


def generate_guards_file(vec: dict, out_path: str):
    lines = [
        "/* Auto-generated interval guards from the VEC (exact doubles; identical to the box used for elimination) */",
        "#include <stdbool.h>",
        "",
    ]
    for regime, feats in vec["feature_intervals"].items():
        lines.append(f"static inline bool guard_{regime}(double realized_vol, double trade_intensity, double avg_trade_size) {{")
        conds = [
            f"({feat} >= {float(c['lower_bound'])!r} && {feat} <= {float(c['upper_bound'])!r})"
            for feat, c in feats.items()
        ]
        lines.append(f"    return {' && '.join(conds)};")
        lines.append("}")
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def generate_dispatch_file(regimes: list, out_path: str):
    lines = [
        "/* Auto-generated dispatch: tries each regime guard in turn, falls back to baseline */",
        '#include "guards.c"',
        "",
        "double predict_baseline(double, double, double);",
    ]
    for r in regimes:
        lines.append(f"double predict_{r}(double, double, double);")
    lines += ["", "double predict_dispatch(double realized_vol, double trade_intensity, double avg_trade_size) {"]
    for r in regimes:
        lines.append(f"    if (guard_{r}(realized_vol, trade_intensity, avg_trade_size)) {{")
        lines.append(f"        return predict_{r}(realized_vol, trade_intensity, avg_trade_size);")
        lines.append("    }")
    lines.append("    return predict_baseline(realized_vol, trade_intensity, avg_trade_size);  /* fallback */")
    lines.append("}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
def validate_risk_firewall(risk_file):
    with open(risk_file, "r") as f:
        source = f.read()

    begin = "PROTECTED_RISK_REGION_BEGIN"
    end = "PROTECTED_RISK_REGION_END"

    if begin not in source or end not in source:
        raise RuntimeError(
            "Risk firewall violation: protected risk region markers missing"
        )

    if source.index(begin) >= source.index(end):
        raise RuntimeError(
            "Risk firewall violation: invalid protected region"
        )

    print("Risk firewall: protected region validated")
def check_risk_dependency(generated_source, risk_file):
    with open(risk_file, "r") as f:
        risk_source = f.read()

    protected_begin = risk_source.index("PROTECTED_RISK_REGION_BEGIN")
    protected_end = risk_source.index("PROTECTED_RISK_REGION_END")

    protected_region = risk_source[protected_begin:protected_end]

    risk_functions = []

    for line in protected_region.splitlines():
        line = line.strip()

        if line.startswith("int ") and "(" in line:
            function_name = line.split("int ", 1)[1].split("(", 1)[0].strip()
            risk_functions.append(function_name)

    for function in risk_functions:
        if function in generated_source:
            raise RuntimeError(
                f"Risk firewall violation: specialized code depends on "
                f"protected risk function '{function}'"
            )

    print("Risk firewall dependency check: PASS")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vec", required=True)
    parser.add_argument("--out_dir", default="generated_c")
    parser.add_argument("--risk_file", default="risk_control.c")
    parser.add_argument(
    "--regenerate",
    nargs="+",
    choices=REGIMES,
    default=REGIMES,
    help="Regimes to regenerate"
)
    parser.add_argument(
    "--regenerate-clauses",
    nargs="+",
    default=[],
    help="Specific VEC-IR clause IDs that triggered regeneration"
)
    parser.add_argument(
    "--inject-risk-test",
    action="store_true",
    help="Inject a deliberate risk dependency for firewall testing"
) 
    args = parser.parse_args()

    if args.regenerate_clauses:
        clause_regimes = {cid.split("-", 1)[0] for cid in args.regenerate_clauses}
        invalid = clause_regimes - set(REGIMES)
        if invalid:
            raise ValueError(f"Unknown clause regime(s): {sorted(invalid)}")
        args.regenerate = sorted(clause_regimes)
        print(f"Clause-level regeneration requested: {args.regenerate_clauses}")
        print(f"Affected regimes: {args.regenerate}")
    os.makedirs(args.out_dir, exist_ok=True)
    validate_risk_firewall(args.risk_file)
    check_risk_dependency("", args.risk_file)

    print(f"Loading model from {args.model} ...")
    model = XGBClassifier()
    model.load_model(args.model)
    trees = load_tree_structure(model)
    base_margin = get_base_margin(model)
    print(f"  {len(trees)} trees, base margin {base_margin!r}")

    print(f"Loading VEC from {args.vec} ...")
    with open(args.vec) as f:
        vec = json.load(f)
    branch_index = index_branch_clauses(vec)
    vec_ir = VECIR.from_vec_json(args.vec)
    print(f"VEC-IR loaded: {len(vec_ir.clauses)} clauses")

    print("Generating baseline ...")
    generate_model_file(trees, None, {}, vec, vec_ir, base_margin, os.path.join(args.out_dir, "baseline_model.c"))

    summary = {}
    for regime in args.regenerate:
        print(f"Generating specialized kernel '{regime}' ...")
        summary[regime] = generate_model_file(
            trees, regime, branch_index, vec, vec_ir, base_margin,
            os.path.join(args.out_dir, f"specialized_{regime}.c"))
    if args.inject_risk_test:
        test_path = os.path.join(
            args.out_dir,
            "specialized_normal.c"
        )

        with open(test_path, "r") as f:
            test_source = f.read()

        test_source += """
/* FIREWALL TEST: deliberate protected-risk dependency */
void firewall_test(void)
{
    risk_check(50.0, 5.0);
}
"""

        with open(test_path, "w") as f:
            f.write(test_source)

        print("Injected deliberate risk dependency into specialized_normal.c")   

    generate_guards_file(vec, os.path.join(args.out_dir, "guards.c"))
    generate_dispatch_file(REGIMES, os.path.join(args.out_dir, "dispatch.c"))
    for regime in args.regenerate:
        specialized_path = os.path.join(
            args.out_dir,
            f"specialized_{regime}.c"
        )

        with open(specialized_path, "r") as f:
            specialized_source = f.read()

        check_risk_dependency(
            specialized_source,
            args.risk_file
        )
    print("\nSummary (Tier 1 exact eliminations / splits kept / Tier 2 hints):")
    for regime, s in summary.items():
        total = s["eliminated"] + s["kept"]
        print(f"  {regime}: {s['eliminated']} eliminated, {s['kept']} kept "
              f"({100.0 * s['eliminated'] / max(total, 1):.1f}% of reachable splits), {s['hinted']} hinted")
    print(f"\nAll files written to {args.out_dir}/")


if __name__ == "__main__":
    main()
