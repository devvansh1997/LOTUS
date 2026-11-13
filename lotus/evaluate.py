"""Evaluation over a set of splits: LOTUS itself, or any detector's scores, with the same metrics."""

import time

import numpy as np

from .detector import Lotus
from .metrics import report


def _require_labels(labels):
    missing = int((np.asarray(labels) < 0).sum())
    if missing:
        raise ValueError("%d responses have no label; evaluation needs a 0/1 label for every response"
                         % missing)


def summarize(rows):
    """Mean, standard deviation (over seeds, ddof=0 as in the paper) and per-seed values."""
    out = {}
    for key in ("pooled_auroc", "within_question_auroc"):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        ok = vals[np.isfinite(vals)]
        out[key] = {"mean": float(ok.mean()) if len(ok) else float("nan"),
                    "std": float(ok.std()) if len(ok) else float("nan"),
                    "per_seed": [float(v) for v in vals]}
    return out


def evaluate_lotus(feats, splits, ranks=None, n_trees=200, n_jobs=-1, log=None):
    """Fit and test LOTUS on every split. The projection is learned from the training rows, then every
    row is transformed in one pass, exactly as the paper's code does, so the results match it."""
    labels, qids = feats.labels, feats.question_ids
    _require_labels(labels)
    rows = []
    for seed, (train, test) in splits:
        if len(np.unique(labels[train])) < 2:
            raise ValueError("seed %d: every training response has the same label, so no detector can "
                             "be trained. The paper protocol trains only on questions with a correct "
                             "response, which fails with one response per question; use the question "
                             "protocol." % seed)
        t0 = time.time()
        det = Lotus(ranks=ranks, n_trees=n_trees, seed=seed, n_jobs=n_jobs)
        det.fit_projection(feats.take(train))
        Z = det.transform(feats)
        det.fit_forest(Z[train], labels[train])
        scores = det.score_transformed(Z[test])
        row = report(scores, labels[test], qids[test])
        row.update(seed=int(seed), n_train=int(len(train)), n_test=int(len(test)),
                   seconds=round(time.time() - t0, 1), test_rows=np.asarray(test), test_scores=scores)
        rows.append(row)
        if log:
            log(row)
    return {"per_seed": rows, "summary": summarize(rows)}


def evaluate_scores(scores, labels, question_ids, splits, log=None):
    """Metrics for another detector's scores on the same splits. scores: (N,) used for every seed, or
    (N, seeds) with one column per seed, for detectors trained on each split."""
    scores = np.asarray(scores, dtype=np.float64)
    labels, qids = np.asarray(labels), np.asarray(question_ids)
    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.ndim != 2 or scores.shape[0] != len(labels):
        raise ValueError("expected one score per response (%d); got an array of shape %s"
                         % (len(labels), scores.shape))
    if scores.shape[1] not in (1, len(splits)):
        raise ValueError("the scores have %d columns; expected 1, or one per seed (%d)"
                         % (scores.shape[1], len(splits)))
    _require_labels(labels)
    rows = []
    for k, (seed, (train, test)) in enumerate(splits):
        s = scores[test, 0 if scores.shape[1] == 1 else k]
        if not np.isfinite(s).all():
            raise ValueError("seed %d: %d test scores are not finite" % (seed, int((~np.isfinite(s)).sum())))
        row = report(s, labels[test], qids[test])
        row.update(seed=int(seed), n_train=int(len(train)), n_test=int(len(test)),
                   test_rows=np.asarray(test), test_scores=s)
        rows.append(row)
        if log:
            log(row)
    return {"per_seed": rows, "summary": summarize(rows)}
