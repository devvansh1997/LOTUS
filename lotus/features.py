"""Features: per-response hidden-state summaries, saved as one .npz file."""

import json

import numpy as np

# The five stored arrays, over a window of P layers and D channels:
#   core      (N, P, D)      peak over the response tokens, per layer and channel
#   q95, q05  (N, P, D)      95th and 5th percentiles over the response tokens
#   v95, v05  (N, P - 1, D)  the same percentiles of the token-wise change between consecutive layers
ARRAYS = ("core", "q95", "q05", "v95", "v05")

# The three summaries LOTUS decomposes, as in the paper: core, static = [q95, q05] and
# velocity = [v95, v05], the last two concatenated along the channel axis.
SUMMARIES = ("core", "static", "velocity")

# Key names in the paper's phase-2 files (written by 42_extract_phase2.py).
PHASE2_KEYS = {"core": "static_max", "q95": "static_q95", "q05": "static_q05",
               "v95": "velocity_q95", "v05": "velocity_q05"}


class Features:
    """Hidden-state summaries of N responses, with their labels and question ids."""

    def __init__(self, arrays, labels=None, question_ids=None, ids=None, n_tokens=None, meta=None):
        missing = [k for k in ARRAYS if k not in arrays]
        if missing:
            raise ValueError("missing arrays: %s" % ", ".join(missing))
        self.arrays = {k: np.asarray(arrays[k]) for k in ARRAYS}
        core = self.arrays["core"]
        if core.ndim != 3:
            raise ValueError("core must be (responses, layers, channels), got shape %s" % (core.shape,))
        n, p, d = core.shape
        expect = {"q95": (n, p, d), "q05": (n, p, d), "v95": (n, p - 1, d), "v05": (n, p - 1, d)}
        for k, shape in expect.items():
            if self.arrays[k].shape != shape:
                raise ValueError("%s has shape %s, expected %s" % (k, self.arrays[k].shape, shape))
        self.labels = (np.full(n, -1, dtype=np.int64) if labels is None
                       else np.asarray(labels, dtype=np.int64))
        self.question_ids = np.arange(n) if question_ids is None else np.asarray(question_ids)
        self.ids = (np.asarray([str(i) for i in range(n)]) if ids is None
                    else np.asarray([str(i) for i in ids]))
        self.n_tokens = None if n_tokens is None else np.asarray(n_tokens, dtype=np.int64)
        self.meta = dict(meta or {})
        for name, col in (("labels", self.labels), ("question_ids", self.question_ids),
                          ("ids", self.ids), ("n_tokens", self.n_tokens)):
            if col is not None and len(col) != n:
                raise ValueError("%s has %d entries for %d responses" % (name, len(col), n))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, rows):
        return self.take(rows)

    @property
    def n_layers(self):
        return self.arrays["core"].shape[1]

    @property
    def hidden_size(self):
        return self.arrays["core"].shape[2]

    @property
    def has_labels(self):
        return bool((self.labels >= 0).all())

    def take(self, rows):
        """The responses at the given row indices, as a new Features."""
        rows = np.asarray(rows, dtype=np.int64)
        return Features({k: v[rows] for k, v in self.arrays.items()}, self.labels[rows],
                        self.question_ids[rows], self.ids[rows],
                        None if self.n_tokens is None else self.n_tokens[rows], self.meta)

    def summary(self, name):
        """One summary as a float32 (N, layers, channels) array, built exactly as the paper's loader."""
        a = self.arrays
        if name == "core":
            return a["core"].astype(np.float32)
        if name == "static":
            return np.concatenate([a["q95"].astype(np.float32), a["q05"].astype(np.float32)], axis=2)
        if name == "velocity":
            return np.concatenate([a["v95"].astype(np.float32), a["v05"].astype(np.float32)], axis=2)
        raise ValueError("unknown summary %r; expected one of %s" % (name, ", ".join(SUMMARIES)))

    def save(self, path):
        """Write one uncompressed .npz file (numpy adds the .npz suffix if it is missing)."""
        extra = {} if self.n_tokens is None else {"n_tokens": self.n_tokens}
        np.savez(path, labels=self.labels, question_ids=self.question_ids, ids=self.ids,
                 meta=np.asarray(json.dumps(self.meta)), **self.arrays, **extra)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if "meta" not in d.files and PHASE2_KEYS["core"] in d.files:
                raise ValueError("%s is one of the paper's phase-2 files; load it with "
                                 "Features.from_phase2_npz" % path)
            missing = [k for k in ARRAYS + ("labels", "question_ids", "ids", "meta") if k not in d.files]
            if missing:
                raise ValueError("%s is not a lotus features file (missing %s)" % (path, ", ".join(missing)))
            return cls({k: d[k] for k in ARRAYS}, d["labels"], d["question_ids"], d["ids"],
                       d["n_tokens"] if "n_tokens" in d.files else None, json.loads(str(d["meta"])))

    @classmethod
    def from_phase2_npz(cls, path):
        """Read one of the paper's phase-2 feature files (blocks 15-23, labels and prompt ids)."""
        with np.load(path, allow_pickle=False) as d:
            arrays = {k: d[v] for k, v in PHASE2_KEYS.items()}
            labels = d["label"].astype(np.int64)
            question_ids = d["prompt_id"].astype(np.int64)
        meta = {"source": "paper phase-2 file", "blocks": list(range(15, 24))}
        return cls(arrays, labels, question_ids, meta=meta)
