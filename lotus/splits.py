"""Train/test splits by question, so no question has responses on both sides.

Splitting by response instead puts most questions on both sides when there are several responses per
question, and inflates the AUROC of every trained detector (by up to 13 points in the paper).
"""

import hashlib
import json

import numpy as np

PAPER_SEEDS = (42, 0, 1, 2, 3)
PROTOCOLS = ("question", "paper")


def _columns(data, labels=None):
    qids = data.question_ids if hasattr(data, "question_ids") else data
    if labels is None and hasattr(data, "labels"):
        labels = data.labels
    return np.asarray(qids), (None if labels is None else np.asarray(labels))


def _rows(codes, train_questions, n_questions):
    is_train = np.zeros(n_questions, dtype=bool)
    is_train[train_questions] = True
    return np.where(is_train[codes])[0], np.where(~is_train[codes])[0]


def question_splits(data, seeds=PAPER_SEEDS, test_size=0.25):
    """Random split of the questions: (1 - test_size) train, the rest test.
    data: Responses, Features, or an array of question ids. Returns [(seed, (train_rows, test_rows))]."""
    qids, _ = _columns(data)
    uniq, codes = np.unique(qids, return_inverse=True)
    n_train = int(len(uniq) * (1 - test_size))
    if n_train < 1 or n_train >= len(uniq):
        raise ValueError("%d questions cannot be split %.0f/%.0f"
                         % (len(uniq), 100 * (1 - test_size), 100 * test_size))
    out = []
    for seed in seeds:
        perm = np.random.RandomState(seed).permutation(len(uniq))
        out.append((int(seed), _rows(codes, perm[:n_train], len(uniq))))
    return out


def paper_splits(data, labels=None, seeds=PAPER_SEEDS, test_size=0.25):
    """The paper's protocol (HARP's, split by question). A question is known if at least one of its
    responses is correct. (1 - test_size) of the known questions train; the other known questions and
    every unknown question test. Needs several responses per question: with one, every training
    response would be correct."""
    qids, labels = _columns(data, labels)
    if labels is None or (labels < 0).any():
        raise ValueError("the paper protocol needs a 0/1 label for every response")
    uniq, codes = np.unique(qids, return_inverse=True)
    known = np.zeros(len(uniq), dtype=bool)
    known[codes[labels == 0]] = True
    out = []
    for seed in seeds:
        known_q = np.where(known)[0].copy()
        np.random.RandomState(seed).shuffle(known_q)
        n_train = int(len(known_q) * (1 - test_size))
        out.append((int(seed), _rows(codes, known_q[:n_train], len(uniq))))
    return out


def make_splits(protocol, data, labels=None, seeds=PAPER_SEEDS, test_size=0.25):
    if protocol == "question":
        return question_splits(data, seeds, test_size)
    if protocol == "paper":
        return paper_splits(data, labels, seeds, test_size)
    raise ValueError("unknown protocol %r; expected one of %s" % (protocol, ", ".join(PROTOCOLS)))


def describe(protocol, test_size, seeds):
    train = 100 * (1 - test_size)
    seeds = " ".join(str(s) for s in seeds)
    if protocol == "paper":
        return ("the paper's protocol: %.0f%% of known questions train; the other known questions and "
                "every unknown question test; seeds %s" % (train, seeds))
    return "by question, %.0f%% train / %.0f%% test, seeds %s" % (train, 100 * test_size, seeds)


def fingerprint(question_ids):
    joined = "\n".join(str(q) for q in np.asarray(question_ids).tolist())
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def save_splits(path, splits, protocol, test_size, question_ids):
    doc = {"tool": "lotus", "protocol": protocol, "test_size": test_size,
           "n_rows": int(len(question_ids)), "question_ids_sha1": fingerprint(question_ids),
           "splits": [{"seed": int(seed), "train": [int(i) for i in tr], "test": [int(i) for i in te]}
                      for seed, (tr, te) in splits]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def load_splits(path, question_ids=None):
    """Returns (splits, document). With question_ids, checks the file was made for the same rows."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if question_ids is not None:
        if doc["n_rows"] != len(question_ids) or doc["question_ids_sha1"] != fingerprint(question_ids):
            raise ValueError("%s was made for different data (%d rows); these data have %d rows"
                             % (path, doc["n_rows"], len(question_ids)))
    splits = [(int(s["seed"]), (np.asarray(s["train"], dtype=np.int64),
                                np.asarray(s["test"], dtype=np.int64))) for s in doc["splits"]]
    return splits, doc
