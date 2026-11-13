"""Pooled and within-question AUROC, as defined in the paper."""

import numpy as np


def pooled_auroc(scores, labels):
    """AUROC over all responses (0.5 = chance). NaN when only one class is present."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def within_question_auroc(scores, labels, question_ids):
    """Probability that a hallucinated response scores above a correct response to the same question,
    over every such pair, ties counting one half. Questions whose responses are all correct or all
    hallucinated contribute no pairs."""
    scores, labels, qids = np.asarray(scores), np.asarray(labels), np.asarray(question_ids)
    order = np.argsort(qids, kind="stable")
    _, starts = np.unique(qids[order], return_index=True)
    bounds = list(starts) + [len(order)]
    concordant, pairs = 0.0, 0
    n_mixed = n_all_correct = n_all_hallucinated = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        s, y = scores[idx], labels[idx]
        h, c = s[y == 1], s[y == 0]
        if len(h) == 0:
            n_all_correct += 1
            continue
        if len(c) == 0:
            n_all_hallucinated += 1
            continue
        n_mixed += 1
        diffs = h[:, None] - c[None, :]
        concordant += (diffs > 0).sum() + 0.5 * (diffs == 0).sum()
        pairs += diffs.size
    return {"auroc": float(concordant / pairs) if pairs else float("nan"),
            "n_mixed_questions": n_mixed, "n_pairs": int(pairs),
            "n_all_correct_questions": n_all_correct, "n_all_hallucinated_questions": n_all_hallucinated}


def report(scores, labels, question_ids):
    """Both metrics in percent, with the counts behind them."""
    labels = np.asarray(labels)
    w = within_question_auroc(scores, labels, question_ids)
    return {"pooled_auroc": 100.0 * pooled_auroc(scores, labels),
            "within_question_auroc": 100.0 * w["auroc"],
            "n": int(len(labels)), "n_hallucinated": int((labels == 1).sum()),
            "n_mixed_questions": w["n_mixed_questions"], "n_pairs": w["n_pairs"]}
