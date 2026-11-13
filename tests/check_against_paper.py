"""Reproduce one of the paper's LOTUS numbers from its cached phase-2 features, with the lotus package.

Run it where the paper's data live, for example:
    python tests/check_against_paper.py \
        --features <data>/qwen-2.5-7b-instruct/truthfulqa_phase2_features.npz --expect 87.83 --paper-code

--expect      the paper's mean pooled AUROC (percent, two decimals) for that model and dataset
--paper-code  also run the original pipeline (44_eval_phase3.py) on the same file and require identical
              splits and per-seed AUROC
Exits 0 when every requested check passes.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lotus import PAPER_SEEDS, Features, evaluate_lotus, paper_splits  # noqa: E402


def load_paper_module(filename):
    spec = importlib.util.spec_from_file_location("paper_" + filename.split("_")[0],
                                                  os.path.join(ROOT, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="Reproduce a LOTUS number from the paper's phase-2 features.")
    ap.add_argument("--features", required=True,
                    help="a phase-2 file: <data>/<model>/<dataset>_phase2_features.npz")
    ap.add_argument("--expect", type=float, help="the paper's mean pooled AUROC (%%), two decimals")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(PAPER_SEEDS))
    ap.add_argument("--paper-code", action="store_true", help="also run 44_eval_phase3.py's pipeline")
    args = ap.parse_args()

    feats = Features.from_phase2_npz(args.features)
    y, q = feats.labels, feats.question_ids
    print("%s: %d responses to %d questions, %.1f%% hallucinated"
          % (args.features, len(y), len(np.unique(q)), 100.0 * y.mean()), flush=True)
    splits = paper_splits(q, y, seeds=args.seeds)

    def log(r):
        print("package     seed %-3d pooled %.4f  within %.4f  (%.0f s)"
              % (r["seed"], r["pooled_auroc"], r["within_question_auroc"], r["seconds"]), flush=True)

    res = evaluate_lotus(feats, splits, log=log)
    s = res["summary"]["pooled_auroc"]
    print("package     mean pooled AUROC %.4f +- %.4f over seeds %s"
          % (s["mean"], s["std"], " ".join(str(x) for x in args.seeds)), flush=True)
    ok = True
    if args.paper_code:
        from sklearn.metrics import roc_auc_score

        m44 = load_paper_module("44_eval_phase3.py")
        a = feats.arrays
        built = m44.build_feats(a["core"], a["q95"], a["q05"], a["v95"], a["v05"])
        builder = m44.make_grouped_builder(m44.CONDITION_SPECS["triple_concat"], built)
        is_known, _ = m44.derive_is_known(y, q)
        for k, seed in enumerate(args.seeds):
            t, v = m44.harp_split(is_known, q, len(y), seed)
            same_split = np.array_equal(t, splits[k][1][0]) and np.array_equal(v, splits[k][1][1])
            Z = builder(t, seed)
            auc = 100.0 * roc_auc_score(y[v], m44.fit_eval("RF", Z[t], y[t], Z[v], seed))
            ours = res["per_seed"][k]["pooled_auroc"]
            same = same_split and auc == ours
            verdict = "identical" if same else "DIFFERENT" + ("" if same_split else " (splits differ)")
            print("paper code  seed %-3d pooled %.4f  package %.4f  %s" % (seed, auc, ours, verdict), flush=True)
            ok = ok and same
    if args.expect is not None:
        got = round(s["mean"], 2)
        match = abs(got - args.expect) < 0.005
        print("expected %.2f, package gives %.2f: %s" % (args.expect, got, "match" if match else "MISMATCH"))
        ok = ok and match
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
