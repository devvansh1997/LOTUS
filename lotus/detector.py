"""The LOTUS detector: robust scaling, a Tucker-2 projection of each summary, and a random forest.

Every computation follows the paper's code line for line (28_eval_band.py for the scaling,
43_eval_phase2.py for the Tucker-2 bases, 26_grouped_baseline.py for the cores and the forest), so the
same features, split and seed give the same AUROC. tests/test_lotus.py checks this.
"""

import numpy as np

from .features import SUMMARIES

# (layer rank, channel rank) per summary: 5*64 + 5*64 + 4*64 = 896 features, as in the paper.
PAPER_RANKS = {"core": (5, 64), "static": (5, 64), "velocity": (4, 64)}

# Rows per chunk when accumulating the layer-mode Gram matrix (the paper's L_MODE_CHUNK).
GRAM_CHUNK = 4096


def fit_robust_scale(X):
    """X: (n, channels), training rows only. Winsorize each channel at its 0.5th and 99.5th
    percentiles, then z-score it by the median and IQR / 1.349 (a robust standard deviation)."""
    p_lo = np.percentile(X, 0.5, axis=0)
    p_hi = np.percentile(X, 99.5, axis=0)
    Xw = np.clip(X, p_lo, p_hi)
    median = np.median(Xw, axis=0)
    q75, q25 = np.percentile(Xw, 75, axis=0), np.percentile(Xw, 25, axis=0)
    iqr = q75 - q25
    scale = iqr / 1.349 + 1e-8
    return {"p_lo": p_lo, "p_hi": p_hi, "median": median, "scale": scale}


def apply_robust_scale(X, params):
    """Clip to the training percentiles, z-score, and clip the result to [-6, 6]."""
    Xw = np.clip(X, params["p_lo"], params["p_hi"])
    Xz = (Xw - params["median"]) / params["scale"]
    return np.clip(Xz, -6, 6)


def tucker2_bases(X, r_layer, r_channel, seed):
    """Tucker-2 bases of a (n, layers, channels) training array, learned without labels.

    Layer basis: the r_layer leading eigenvectors of the layer-mode Gram matrix, sum_n X_n X_n^T.
    Channel basis: the r_channel leading right singular vectors of X with responses and layers
    stacked as rows, by randomized SVD with the given seed."""
    from sklearn.utils.extmath import randomized_svd

    n, L, D = X.shape
    if not (0 < r_layer <= L and 0 < r_channel <= D):
        raise ValueError("ranks (%d, %d) do not fit a summary of %d layers and %d channels"
                         % (r_layer, r_channel, L, D))
    if n * L < r_channel:
        raise ValueError("%d training responses are too few for a channel rank of %d" % (n, r_channel))
    A = np.zeros((L, L), dtype=np.float64)
    for start in range(0, n, GRAM_CHUNK):
        chunk = X[start:start + GRAM_CHUNK].astype(np.float64)
        flat = chunk.transpose(1, 0, 2).reshape(L, -1)
        A += flat @ flat.T
    _, vecs = np.linalg.eigh(A)
    U_L = np.flip(vecs[:, -r_layer:], axis=1).copy()
    merged = X.reshape(n * L, D).astype(np.float32)
    _, _, Vt = randomized_svd(merged, n_components=r_channel, random_state=seed)
    return U_L.astype(np.float32), Vt.T.astype(np.float32)


def project_core(X, U_L, U_D):
    """(n, layers, channels) -> (n, r_layer * r_channel): each response's core U_L^T X_n U_D, flattened."""
    temp = X.astype(np.float32) @ U_D
    G = np.einsum("nld,lr->nrd", temp, U_L)
    return G.reshape(G.shape[0], -1)


class Lotus:
    """LOTUS with the paper's defaults: ranks 5x64, 5x64 and 4x64 (896 features) and a 200-tree,
    class-balanced random forest. Everything before the forest is learned without labels.

    fit(feats, labels) and score(feats) are all most uses need. evaluate_lotus() uses the lower-level
    fit_projection / transform / fit_forest steps to reproduce the paper's order of operations."""

    def __init__(self, ranks=None, n_trees=200, seed=0, n_jobs=-1):
        self.ranks = dict(PAPER_RANKS if ranks is None else ranks)
        missing = [s for s in SUMMARIES if s not in self.ranks]
        if missing:
            raise ValueError("ranks missing for %s" % ", ".join(missing))
        self.n_trees = int(n_trees)
        self.seed = int(seed)
        self.n_jobs = n_jobs

    @property
    def n_features(self):
        return int(sum(self.ranks[s][0] * self.ranks[s][1] for s in SUMMARIES))

    def fit_projection(self, feats):
        """Learn the scaling and the Tucker-2 bases of each summary from these (training) responses.
        Labels are not used. Summary i uses seed + i, as in the paper."""
        self.scalers_, self.bases_ = {}, {}
        for i, name in enumerate(SUMMARIES):
            X = feats.summary(name)
            n, L, D = X.shape
            params = fit_robust_scale(X.reshape(n, L * D))
            Xs = apply_robust_scale(X.reshape(n, L * D), params).reshape(n, L, D)
            r_layer, r_channel = self.ranks[name]
            self.scalers_[name] = params
            self.bases_[name] = tucker2_bases(Xs, r_layer, r_channel, self.seed + i)
        self.trained_on_ = {"n_layers": int(feats.n_layers), "hidden_size": int(feats.hidden_size),
                            "blocks": feats.meta.get("blocks"), "model": feats.meta.get("model")}
        return self

    def transform(self, feats):
        """(n, n_features) Tucker-2 cores of every response, in the order core, static, velocity."""
        if not hasattr(self, "bases_"):
            raise ValueError("fit the detector first")
        seen = self.trained_on_
        if (feats.n_layers, feats.hidden_size) != (seen["n_layers"], seen["hidden_size"]):
            raise ValueError("these features have %d layers x %d channels; the detector was trained on "
                             "%d x %d" % (feats.n_layers, feats.hidden_size, seen["n_layers"],
                                          seen["hidden_size"]))
        parts = []
        for name in SUMMARIES:
            X = feats.summary(name)
            n, L, D = X.shape
            Xs = apply_robust_scale(X.reshape(n, L * D), self.scalers_[name]).reshape(n, L, D)
            parts.append(project_core(Xs, *self.bases_[name]))
        return np.concatenate(parts, axis=1)

    def fit_forest(self, Z, labels):
        """Train the random forest, the only step that uses labels."""
        from sklearn.ensemble import RandomForestClassifier

        labels = np.asarray(labels)
        classes = np.unique(labels)
        if not np.array_equal(classes, [0, 1]):
            raise ValueError("training labels must include both 0 (correct) and 1 (hallucinated); "
                             "got %s" % classes.tolist())
        self.forest_ = RandomForestClassifier(n_estimators=self.n_trees, class_weight="balanced",
                                              random_state=self.seed, n_jobs=self.n_jobs)
        self.forest_.fit(Z, labels)
        return self

    def fit(self, feats, labels=None):
        """Fit on these responses: projection without labels, then the forest with them."""
        labels = feats.labels if labels is None else np.asarray(labels)
        self.fit_projection(feats)
        return self.fit_forest(self.transform(feats), labels)

    def score_transformed(self, Z):
        return self.forest_.predict_proba(Z)[:, 1]

    def score(self, feats):
        """Probability that each response is hallucinated (higher = more likely)."""
        return self.score_transformed(self.transform(feats))

    def save(self, path):
        import joblib
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path):
        import joblib
        det = joblib.load(path)
        if not isinstance(det, Lotus):
            raise ValueError("%s does not hold a Lotus detector" % path)
        return det
