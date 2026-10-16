"""
gen_flat_baseline.py -- strong, non-specialized baseline: a branchless flat-array tree evaluator.

Each tree is padded to a perfect binary tree of depth D (D = deepest tree). Evaluation of one tree is
D iterations of   idx = 2*idx + 1 + (x[feat[idx]] >= thr[idx])   with no data-dependent branches.
Early leaves are padded with pass-through nodes (thr = +inf) and replicated leaf values.
Semantics are identical to the if/else baseline (yes = x < t, same float32 thresholds, same summation
order), so predict_flat must equal predict_baseline exactly.

Usage:
    python gen_flat_baseline.py --model model_big.json --out_dir generated_c_big
Writes <out_dir>/flat_model.c  (defines double predict_flat(double,double,double))
"""
import argparse
import os
from xgboost import XGBClassifier
from generate_specialized_code import FEATURE_COLS, f32, get_base_margin, load_tree_structure


def tree_depth(node) -> int:
    return 0 if "leaf" in node else 1 + max(tree_depth(c) for c in node["children"])


def feat_index(name: str) -> int:
    if name in FEATURE_COLS:
        return FEATURE_COLS.index(name)
    if name[:1] == "f" and name[1:].isdigit():
        return int(name[1:])
    raise ValueError(f"unknown feature {name}")


def build_arrays(tree, D):
    n_int, n_leaf = 2 ** D - 1, 2 ** D
    feat = [0] * n_int
    thr = ["INFINITY"] * n_int
    leaf = [0.0] * n_leaf

    def fill_const(pos, d, val):
        stack = [(pos, d)]
        while stack:
            p, dd = stack.pop()
            if dd == D:
                leaf[p - n_int] = val
            else:
                stack.append((2 * p + 1, dd + 1))
                stack.append((2 * p + 2, dd + 1))

    def walk(node, pos, d):
        if "leaf" in node:
            fill_const(pos, d, f32(node["leaf"]))
            return
        feat[pos] = feat_index(node["split"])
        thr[pos] = repr(f32(node["split_condition"]))
        yes = next(c for c in node["children"] if c["nodeid"] == node["yes"])
        no = next(c for c in node["children"] if c["nodeid"] == node["no"])
        walk(yes, 2 * pos + 1, d + 1)
        walk(no, 2 * pos + 2, d + 1)

    walk(tree, 0, 0)
    return feat, thr, leaf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", default="generated_c")
    a = ap.parse_args()

    model = XGBClassifier()
    model.load_model(a.model)
    trees = load_tree_structure(model)
    base_margin = get_base_margin(model)
    D = max(tree_depth(t) for t in trees)
    T, NI = len(trees), 2 ** D - 1
    print(f"{T} trees, padded depth D={D}, base margin {base_margin!r}")

    lines = ["/* Auto-generated flat branchless baseline */", "#include <math.h>", "",
             f"#define NT {T}", f"#define DEPTH {D}", f"#define NI {NI}", ""]
    feats, thrs, leaves = [], [], []
    for t in trees:
        f, th, lf = build_arrays(t, D)
        feats.append("{" + ",".join(map(str, f)) + "}")
        thrs.append("{" + ",".join(th) + "}")
        leaves.append("{" + ",".join(repr(v) for v in lf) + "}")
    lines.append(f"static const unsigned char FEAT[NT][NI] = {{{','.join(feats)}}};")
    lines.append(f"static const double THR[NT][NI] = {{{','.join(thrs)}}};")
    lines.append(f"static const double LEAF[NT][{2 ** D}] = {{{','.join(leaves)}}};")
    lines += ["",
              "double predict_flat(double realized_vol, double trade_intensity, double avg_trade_size) {",
              "    const double x[3] = {realized_vol, trade_intensity, avg_trade_size};",
              f"    double sum = {base_margin!r};",
              "    for (int t = 0; t < NT; t++) {",
              "        unsigned idx = 0;",
              "        for (int d = 0; d < DEPTH; d++) idx = 2 * idx + 1 + (x[FEAT[t][idx]] >= THR[t][idx]);",
              "        sum += LEAF[t][idx - NI];",
              "    }",
              "    return 1.0 / (1.0 + exp(-sum));",
              "}", ""]
    path = os.path.join(a.out_dir, "flat_model.c")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
