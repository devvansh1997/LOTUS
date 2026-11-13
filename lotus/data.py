"""Responses: the rows LOTUS reads, one per generated response, loaded from JSONL or built in Python."""

import json

import numpy as np

REQUIRED = ("question_id", "prompt", "response")


def _label(value, where):
    """1 = hallucinated (wrong), 0 = correct, -1 = no label (allowed only for scoring)."""
    if value is None:
        return -1
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1, -1):
        return int(value)
    raise ValueError("%s: label must be 0 (correct) or 1 (hallucinated), got %r" % (where, value))


def _question_ids(values):
    """Integer ids stay integers, so they sort numerically; anything else becomes a string."""
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in values):
        return np.asarray(values, dtype=np.int64)
    return np.asarray([str(v) for v in values])


class Responses:
    """Prompts, responses and labels, one row per response.

    question_ids  responses to the same question share an id; splits never separate them
    prompts       the exact text the model was given (chat template included, if one was used)
    responses     the generated text
    labels        1 = hallucinated (wrong), 0 = correct, -1 = no label (allowed only for scoring)
    ids           optional row identifiers, carried through to the outputs
    """

    def __init__(self, question_ids, prompts, responses, labels=None, ids=None):
        n = len(prompts)
        if n == 0:
            raise ValueError("no responses")
        for name, col in (("question_ids", question_ids), ("responses", responses),
                          ("labels", labels), ("ids", ids)):
            if col is not None and len(col) != n:
                raise ValueError("%s has %d entries, prompts has %d" % (name, len(col), n))
        self.question_ids = _question_ids(list(question_ids))
        self.prompts = [str(p) for p in prompts]
        self.responses = ["" if r is None else str(r) for r in responses]
        labels = [None] * n if labels is None else list(labels)
        self.labels = np.asarray([_label(v, "row %d" % (i + 1)) for i, v in enumerate(labels)],
                                 dtype=np.int64)
        self.ids = np.asarray([str(v) for v in (range(n) if ids is None else ids)])

    def __len__(self):
        return len(self.prompts)

    @property
    def has_labels(self):
        return bool((self.labels >= 0).all())

    def take(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        return Responses(self.question_ids[rows].tolist(), [self.prompts[i] for i in rows],
                         [self.responses[i] for i in rows], self.labels[rows].tolist(),
                         self.ids[rows].tolist())

    @classmethod
    def from_records(cls, records, source="record"):
        """records: an iterable of dicts with question_id, prompt, response and optional label, id."""
        cols = {"question_id": [], "prompt": [], "response": [], "label": [], "id": []}
        for i, rec in enumerate(records):
            where = "%s %d" % (source, i + 1)
            cls._add(cols, rec, where, default_id=i)
        return cls._build(cols)

    @classmethod
    def from_jsonl(cls, path):
        """One JSON object per line; blank lines are skipped. Errors name the offending line."""
        cols = {"question_id": [], "prompt": [], "response": [], "label": [], "id": []}
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                where = "%s line %d" % (path, line_no)
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError("%s: not valid JSON (%s)" % (where, e))
                cls._add(cols, rec, where, default_id=len(cols["prompt"]))
        if not cols["prompt"]:
            raise ValueError("%s has no rows" % path)
        return cls._build(cols)

    @staticmethod
    def _add(cols, rec, where, default_id):
        if not isinstance(rec, dict):
            raise ValueError("%s: expected a JSON object" % where)
        missing = [k for k in REQUIRED if k not in rec]
        if missing:
            raise ValueError("%s: missing field(s) %s" % (where, ", ".join(missing)))
        for k in ("prompt", "response"):
            if rec[k] is not None and not isinstance(rec[k], str):
                raise ValueError("%s: %s must be a string" % (where, k))
        cols["question_id"].append(rec["question_id"])
        cols["prompt"].append(rec["prompt"])
        cols["response"].append(rec["response"])
        cols["label"].append(_label(rec.get("label"), where))
        cols["id"].append(rec.get("id", default_id))

    @classmethod
    def _build(cls, cols):
        return cls(cols["question_id"], cols["prompt"], cols["response"], cols["label"], cols["id"])
