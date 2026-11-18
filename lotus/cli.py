"""Command-line tool: lotus extract | evaluate | splits | fit | score. Run `lotus <command> -h`."""

import argparse
import io
import json
import os
import sys

import numpy as np

from ._version import __version__
from .splits import PAPER_SEEDS, PROTOCOLS


def _n(x):
    return "{:,}".format(int(x))


def _pct(x):
    return "%.2f" % x if np.isfinite(x) else "n/a"


def _size(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return "%.1f %s" % (nbytes, unit)
        nbytes /= 1024.0


def _describe(labels, question_ids):
    labels = np.asarray(labels)
    text = "%s responses to %s questions" % (_n(len(labels)), _n(len(np.unique(question_ids))))
    known = labels >= 0
    if known.any():
        text += ", %.1f%% labelled hallucinated" % (100.0 * (labels[known] == 1).mean())
    if not known.all():
        text += " (%s without a label)" % _n((~known).sum())
    return text


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def _load_scores(path):
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    with open(path, encoding="utf-8") as f:
        text = f.read().replace(",", " ")
    return np.loadtxt(io.StringIO(text), dtype=np.float64, ndmin=1)


def _print_table(summary):
    print("%-20s %7s %6s   %s" % ("", "mean", "std", "per seed"))
    for key, label in (("pooled_auroc", "pooled AUROC (%)"), ("within_question_auroc", "within-question (%)")):
        s = summary[key]
        if not np.isfinite(s["mean"]):
            print("%-20s n/a (no test question has both correct and hallucinated responses)" % label)
            continue
        print("%-20s %7.2f %6.2f   %s" % (label, s["mean"], s["std"], " ".join(_pct(v) for v in s["per_seed"])))


def _log_seed(r):
    took = "  %.0f s" % r["seconds"] if "seconds" in r else ""
    print("seed %-4d pooled %6s  within %6s  train %s / test %s responses%s"
          % (r["seed"], _pct(r["pooled_auroc"]), _pct(r["within_question_auroc"]), _n(r["n_train"]),
             _n(r["n_test"]), took), flush=True)


def cmd_extract(args):
    from .data import Responses
    from .extraction import (DEFAULT_BLOCKS, blocks_text, extract, load_model, parse_blocks,
                             resolve_device, text_config)

    data = Responses.from_jsonl(args.data)
    if args.limit:
        data = data.take(np.arange(min(args.limit, len(data))))
    blocks = parse_blocks(args.layers)
    model, tok = load_model(args.model, device=resolve_device(args.device), dtype=args.dtype,
                            trust_remote_code=args.trust_remote_code)
    cfg = text_config(model.config)
    print("model    %s: %d blocks, hidden size %d" % (args.model, cfg.num_hidden_layers, cfg.hidden_size))
    note = " (the paper's window; change with --layers)" if blocks == DEFAULT_BLOCKS else ""
    print("window   blocks %s%s" % (blocks_text(blocks), note))
    print("data     " + _describe(data.labels, data.question_ids))
    feats = extract(model, data, tokenizer=tok, blocks=blocks, batch_size=args.batch_size,
                    append_eos=not args.no_eos, progress=True)
    feats.meta["model"] = args.model
    out = args.out if args.out.endswith(".npz") else args.out + ".npz"
    feats.save(out)
    print("saved    %s: %d x %d values per response, %s, %s"
          % (out, 5 * len(blocks) - 2, feats.hidden_size, _size(os.path.getsize(out)), feats.meta["storage"]))
    return 0


def _get_splits(args, labels, question_ids):
    from .splits import load_splits, make_splits

    if args.splits:
        splits, doc = load_splits(args.splits, question_ids)
        return splits, doc["protocol"], doc["test_size"]
    return make_splits(args.protocol, question_ids, labels, args.seeds, args.test_size), args.protocol, args.test_size


def cmd_evaluate(args):
    from .data import Responses
    from .evaluate import evaluate_lotus, evaluate_scores
    from .features import Features
    from .splits import describe

    if bool(args.features) == bool(args.scores):
        raise ValueError("give either --features (to evaluate LOTUS) or --scores with --data "
                         "(to evaluate another detector)")
    if args.features:
        feats = Features.load(args.features)
        labels, qids, ids = feats.labels, feats.question_ids, feats.ids
        print("LOTUS on %s: %s" % (args.features, _describe(labels, qids)))
    else:
        if not args.data:
            raise ValueError("--scores needs --data, the JSONL file the scores belong to")
        data = Responses.from_jsonl(args.data)
        labels, qids, ids = data.labels, data.question_ids, data.ids
        scores = _load_scores(args.scores)
        print("scores in %s: %s" % (args.scores, _describe(labels, qids)))
    splits, protocol, test_size = _get_splits(args, labels, qids)
    origin = " (from %s)" % args.splits if args.splits else ""
    print("split    %s%s" % (describe(protocol, test_size, [s for s, _ in splits]), origin))
    if args.features:
        res = evaluate_lotus(feats, splits, log=_log_seed)
        default_out = os.path.splitext(args.features)[0] + ".lotus.json"
    else:
        res = evaluate_scores(scores, labels, qids, splits, log=_log_seed)
        default_out = os.path.splitext(args.scores)[0] + ".eval.json"
    _print_table(res["summary"])
    out = args.out or default_out
    doc = {"lotus_version": __version__, "source": args.features or args.scores, "protocol": protocol,
           "test_size": test_size, "seeds": [s for s, _ in splits], "summary": res["summary"],
           "per_seed": [dict(r, test_ids=ids[r["test_rows"]]) for r in res["per_seed"]]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_jsonable(doc), f)
    print("results  %s (per-seed metrics and per-response test scores)" % out)
    return 0


def cmd_splits(args):
    from .data import Responses
    from .features import Features
    from .splits import describe, make_splits, save_splits

    if bool(args.data) == bool(args.features):
        raise ValueError("give either --data (JSONL) or --features")
    src = Responses.from_jsonl(args.data) if args.data else Features.load(args.features)
    labels, qids = src.labels, src.question_ids
    splits = make_splits(args.protocol, qids, labels, args.seeds, args.test_size)
    save_splits(args.out, splits, args.protocol, args.test_size, qids)
    print("split    " + describe(args.protocol, args.test_size, args.seeds))
    for seed, (tr, te) in splits:
        print("seed %-4d train %s responses (%s questions), test %s responses (%s questions)"
              % (seed, _n(len(tr)), _n(len(np.unique(qids[tr]))), _n(len(te)), _n(len(np.unique(qids[te])))))
    print("saved    %s (row numbers follow the order of the input)" % args.out)
    return 0


def cmd_fit(args):
    from .detector import Lotus
    from .features import Features

    feats = Features.load(args.features)
    if not feats.has_labels:
        raise ValueError("every response needs a 0/1 label to train the detector")
    print("training on %s: %s" % (args.features, _describe(feats.labels, feats.question_ids)))
    det = Lotus(seed=args.seed).fit(feats)
    det.save(args.out)
    print("saved    %s (%d features, %d trees)" % (args.out, det.n_features, det.n_trees))
    return 0


def cmd_score(args):
    from .detector import Lotus
    from .features import Features
    from .metrics import report

    det = Lotus.load(args.detector)
    feats = Features.load(args.features)
    seen = det.trained_on_
    for key in ("model", "blocks"):
        if seen.get(key) is not None and feats.meta.get(key) is not None and seen[key] != feats.meta[key]:
            print("warning: the detector was trained on %s %s; these features have %s"
                  % (key, seen[key], feats.meta[key]))
    scores = det.score(feats)
    np.savetxt(args.out, scores, fmt="%.6f")
    print("scored   %s responses -> %s (probability of hallucination, one line per response)"
          % (_n(len(scores)), args.out))
    if feats.has_labels and len(np.unique(feats.labels)) == 2:
        r = report(scores, feats.labels, feats.question_ids)
        print("these responses have labels: pooled AUROC %s, within-question %s"
              % (_pct(r["pooled_auroc"]), _pct(r["within_question_auroc"])))
        questions = np.unique(feats.question_ids).tolist()
        seen_before = len(set(questions) & set(seen.get("questions", [])))
        if seen_before:
            print("warning: %d of these %d questions were used to train this detector, so these AUROCs "
                  "are inflated; use lotus evaluate for a fair estimate" % (seen_before, len(questions)))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="lotus", description="LOTUS hallucination detection: extract "
                                "hidden-state summaries once, then fit and evaluate the detector.")
    p.add_argument("--version", action="version", version="lotus " + __version__)
    sub = p.add_subparsers(dest="command", metavar="command")
    sub.required = True

    e = sub.add_parser("extract", help="read hidden-state summaries of each response (GPU)")
    e.add_argument("--model", required=True, help="Hugging Face model id or local directory")
    e.add_argument("--data", required=True, help="JSONL with question_id, prompt, response, label")
    e.add_argument("--out", required=True, help="features file to write (.npz)")
    e.add_argument("--layers", default="15-23",
                   help="blocks to read, 0-based: a range such as 15-23 (default, the paper's window) "
                        "or a list such as 10,12,14")
    e.add_argument("--batch-size", type=int, default=4)
    e.add_argument("--device", default=None,
                   help="cuda (default when available), cpu, cuda:1, or auto to spread the model "
                        "over several GPUs (needs accelerate)")
    e.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    e.add_argument("--no-eos", action="store_true",
                   help="do not append the end-of-sequence token to each response")
    e.add_argument("--trust-remote-code", action="store_true")
    e.add_argument("--limit", type=int, default=0, help="read only the first N responses (a quick try)")
    e.set_defaults(func=cmd_extract)

    def split_args(a):
        a.add_argument("--protocol", default="question", choices=list(PROTOCOLS),
                       help="question (default): random split by question; paper: the paper's "
                            "protocol, which tests on every question with no correct response")
        a.add_argument("--seeds", type=int, nargs="+", default=list(PAPER_SEEDS))
        a.add_argument("--test-size", type=float, default=0.25)

    v = sub.add_parser("evaluate", help="fit and test LOTUS on each split, or evaluate another "
                                        "detector's scores on the same splits")
    v.add_argument("--features", help="features file from lotus extract")
    v.add_argument("--scores", help="another detector's scores: one line per response (one column, "
                                    "or one column per seed), or an .npy array")
    v.add_argument("--data", help="with --scores: the JSONL file the scores belong to")
    v.add_argument("--splits", help="splits file from lotus splits (replaces --protocol, --seeds)")
    split_args(v)
    v.add_argument("--out", help="results file (default: next to the input, .lotus.json or .eval.json)")
    v.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("splits", help="write the train and test rows of each seed, for other detectors")
    s.add_argument("--data", help="JSONL file")
    s.add_argument("--features", help="or a features file")
    split_args(s)
    s.add_argument("--out", required=True, help="splits file to write (.json)")
    s.set_defaults(func=cmd_splits)

    f = sub.add_parser("fit", help="train LOTUS on every labelled response and save it")
    f.add_argument("--features", required=True)
    f.add_argument("--out", required=True, help="detector file to write (.joblib)")
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=cmd_fit)

    c = sub.add_parser("score", help="score responses with a saved detector")
    c.add_argument("--detector", required=True)
    c.add_argument("--features", required=True, help="features of the responses to score")
    c.add_argument("--out", required=True, help="text file, one score per line")
    c.set_defaults(func=cmd_score)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (ValueError, OSError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
