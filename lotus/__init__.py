"""LOTUS: hallucination detection from order statistics of an LLM's hidden states.

Typical use:
    lotus extract  --model <model> --data responses.jsonl --out features.npz   (GPU, once)
    lotus evaluate --features features.npz                                      (CPU)

The same steps from Python:
    from lotus import Responses, extract, Lotus, question_splits, report

See README.md for the data format and every command.
"""

from ._version import __version__
from .data import Responses
from .detector import PAPER_RANKS, Lotus
from .evaluate import evaluate_lotus, evaluate_scores
from .extraction import DEFAULT_BLOCKS, extract, load_model
from .features import ARRAYS, SUMMARIES, Features
from .metrics import pooled_auroc, report, within_question_auroc
from .splits import PAPER_SEEDS, load_splits, make_splits, paper_splits, question_splits, save_splits

__all__ = [
    "__version__", "Responses", "Features", "Lotus", "extract", "load_model",
    "question_splits", "paper_splits", "make_splits", "save_splits", "load_splits",
    "evaluate_lotus", "evaluate_scores", "pooled_auroc", "within_question_auroc", "report",
    "ARRAYS", "SUMMARIES", "DEFAULT_BLOCKS", "PAPER_RANKS", "PAPER_SEEDS",
]
