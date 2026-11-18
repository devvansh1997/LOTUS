# LLM Hallucination Detection via Low-rank Multilinear Decomposition

Code for the paper *LLM Hallucination Detection via Low-rank Multilinear Decomposition*, which
introduces LOTUS (low-rank order-statistic Tucker subspaces).

LOTUS summarizes the residual stream of a generated response by order statistics
over its tokens, compresses the resulting layer-by-feature matrices with a Tucker-2 decomposition whose
two bases are fitted to unlabeled training activations, and trains a classifier on the concatenated
cores.

This repository has two parts:

- **The `lotus` package**, to run LOTUS on your own model and data and benchmark it against other
  detectors: see [Use LOTUS on your own model](#use-lotus-on-your-own-model).
- **The paper's pipeline**: the generation, extraction, evaluation and analysis code, plus the cluster
  job scripts used to run it: see [Reproducing the paper](#reproducing-the-paper).

No data, model weights, logs or result files are included.

## Use LOTUS on your own model

LOTUS needs an open-weight model (it reads the model's hidden states), the prompts you gave the model,
its responses, and a label saying whether each response is correct. Extraction runs once on a GPU;
fitting and evaluation run on a CPU.

### Install

Use a fresh environment with Python 3.10 or later:

```bash
conda create -n lotus python=3.10 -y
conda activate lotus
```

Install PyTorch for your CUDA version (https://pytorch.org), then, from a clone of this repository:

```bash
pip install -e .
```

This installs the `lotus` command and the `lotus` Python package. `lotus <command> -h` lists every
option.

### 1. Prepare your data

One JSON object per line, one line per response:

```json
{"question_id": "q0412", "prompt": "Q: Which vitamin deficiency causes scurvy?\nA:", "response": " Vitamin C", "label": 0}
```

| Field | Meaning |
|---|---|
| `question_id` | Shared by every response to the same question. Splits never separate them, and within-question AUROC compares them. |
| `prompt` | The exact text the model was given, chat template included if you used one. |
| `response` | The generated text. |
| `label` | 1 if the response is wrong (hallucinated), 0 if it is correct. Not needed by `lotus score`. |
| `id` | Optional; carried through to the outputs. |

LOTUS does not judge correctness, so label the responses however suits your task: exact match for
multiple choice, a judge model or annotators for free text. Several sampled responses per question (the
paper uses 10) give within-question AUROC and allow the paper's protocol; one response per question
works with the default split.

### 2. Extract hidden-state summaries (GPU)

```bash
lotus extract --model <model id or path> --data responses.jsonl --out features.npz
```

```text
model    meta-llama/Llama-3.1-8B-Instruct: 32 blocks, hidden size 4096
window   blocks 15-23 (the paper's window; change with --layers)
data     12,730 responses to 1,273 questions, 38.2% labelled hallucinated
reading: 100%|##########| 3183/3183 [18:40<00:00, 2.84batch/s]
saved    features.npz: 43 x 4096 values per response, 4.2 GB, float16
```

LOTUS runs one forward pass over each prompt and response; it generates nothing. At each block of the
window it keeps, over the response's tokens, the peak, the 95th and 5th percentiles, and the same two
percentiles of the change from the previous block: 43 values per hidden dimension with the default
window, stored as float16. That is about 350 KB per response for a hidden size of 4,096, and the whole
file is held in memory while it is written.

- `--layers` (default `15-23`): the blocks to read, 0-based. The paper used blocks 15-23 for models
  with 28, 32 and 44 blocks; a model with fewer than 24 blocks needs another window.
- The end-of-sequence token is appended to every response, as in the paper, whose answer window runs
  through the stop token. It also gives an empty response a token to read. `--no-eos` turns it off.
- `--batch-size` (default 4), `--device` (`cuda`, `cpu`, or `auto` to spread a large model over
  several GPUs, which needs `accelerate`), `--dtype` (default bfloat16 on a GPU that supports it),
  `--trust-remote-code`, and `--limit N` to try the first N responses.

### 3. Evaluate LOTUS (CPU)

```bash
lotus evaluate --features features.npz
```

```text
LOTUS on features.npz: 12,730 responses to 1,273 questions, 38.2% labelled hallucinated
split    by question, 75% train / 25% test, seeds 42 0 1 2 3
seed 42   pooled  83.50  within  69.90  train 9,547 / test 3,183 responses  151 s
seed 0    pooled  85.04  within  73.60  train 9,555 / test 3,175 responses  149 s
seed 1    pooled  84.77  within  70.20  train 9,541 / test 3,189 responses  150 s
seed 2    pooled  83.12  within  72.90  train 9,549 / test 3,181 responses  152 s
seed 3    pooled  84.19  within  69.90  train 9,546 / test 3,184 responses  148 s
                        mean    std   per seed
pooled AUROC (%)       84.12   0.73   83.50 85.04 84.77 83.12 84.19
within-question (%)    71.30   1.61   69.90 73.60 70.20 72.90 69.90
results  features.lotus.json (per-seed metrics and per-response test scores)
```

The numbers above are illustrative; yours depend on the model and the data.

For each seed, LOTUS learns from the training responses, without labels, a robust scaling of every
layer and channel and a Tucker-2 projection of each summary (ranks 5x64, 5x64 and 4x64, so 896
features). It then trains a 200-tree class-balanced random forest on the training labels and scores
the test responses. In the paper, one fit took 2.4-2.6 minutes on a CPU with 3,800-5,000 training
responses, and scoring took about 1.5 ms per response.

- **Pooled AUROC**: AUROC over all test responses.
- **Within-question AUROC**: the probability that a hallucinated response scores above a correct
  response to the same question, over every such pair, ties counting one half. Only questions with
  both kinds of response contribute.
- `--protocol question` (default): a random split of the questions, 75% train and 25% test.
- `--protocol paper`: the paper's protocol. A question is known if at least one of its responses is
  correct; 75% of the known questions train, and the other known questions and every unknown question
  test. It needs several responses per question: with one, every training response would be correct.
- `--seeds` (default `42 0 1 2 3`, the paper's) and `--test-size` (default 0.25).

Do not split by response when there are several responses per question. Most questions then land on
both sides, which inflates the AUROC of every trained detector (by up to 13 points in the paper).

### 4. Compare with other detectors on the same splits

```bash
lotus splits   --data responses.jsonl --out splits.json
lotus evaluate --scores other_detector.txt --data responses.jsonl --splits splits.json
```

`splits.json` lists the training and test rows of each seed (0-based, in the order of the JSONL file);
use it to train any supervised baseline. A score file has one line per response, in the same order.
One column is used for every seed; several columns give one column per seed, for detectors trained on
each split. An `.npy` array of shape (N,) or (N, seeds) also works. Both commands report the same
metrics, and `lotus evaluate --features features.npz --splits splits.json` evaluates LOTUS on exactly
the same rows.

### Train once, score new responses

```bash
lotus fit     --features features.npz --out lotus.joblib
lotus extract --model <same model> --data new.jsonl --out new.npz
lotus score   --detector lotus.joblib --features new.npz --out new_scores.txt
```

Each line of `new_scores.txt` is the probability that the response is hallucinated.

### From Python

```python
from lotus import Responses, extract, Lotus, question_splits, report

data = Responses.from_jsonl("responses.jsonl")
feats = extract("<model id or path>", data)     # or extract(model, data, tokenizer=tokenizer)
for seed, (train, test) in question_splits(data, seeds=[42, 0, 1, 2, 3]):
    det = Lotus(seed=seed).fit(feats[train], data.labels[train])
    scores = det.score(feats[test])            # higher = more likely hallucinated
    print(seed, report(scores, data.labels[test], data.question_ids[test]))
```

`evaluate_lotus(feats, splits)` runs the same evaluation as `lotus evaluate`.

### Tests

```bash
python tests/test_lotus.py
```

The tests check that the package reproduces this repository's pipeline exactly on synthetic data: the
scaling, the Tucker-2 bases and cores, the splits, both metrics, the token pooling and the end-to-end
AUROC. They also run the extraction and every command on a small randomly initialised model, so they
need no GPU, data or downloads. `tests/check_against_paper.py` reproduces a number from the paper from
its cached features.

## Reproducing the paper

### Layout

```
39_generate_dataset.py      generate K responses per question and label them
40_validate_dataset.py      integrity checks on a generated dataset
42_extract_phase2.py        summaries over the layer window (peak, range, cross-layer difference)
44_eval_phase3.py           question-level and answer-level evaluation of the detector
56_run_method.py + methods/ the baseline detectors, one file per method
49_harp_adapter.py          convert our generations into HARP's input format
50_harp_reeval_prompt_level.py  re-run HARP under both splits, pooled and within-question
51_judge_agreement.py       compare our labels with the released judge
57..64_*.py                 ablations: all-layer extraction, layer sweep, window sweep, rank sweep,
                            reductions (flat PCA, random projection), Tucker-3, ACT-ViT tensors
65_*, 66_cross_model.py     cross-model transfer with a label-free ridge alignment
60_class_spectrum.py        class-wise covariance spectra
config.yaml                 models, datasets and generation settings
slurm/                      one job script per stage
submit_*.sh                 submission wrappers; each documents its own cost and dependencies
lotus/, pyproject.toml      the installable package (see "Use LOTUS on your own model")
tests/                      tests of the package against this pipeline
```

Scripts are numbered by the order in which they entered the project, not by pipeline order; each file
begins with a docstring stating what it does, what it reads and what it writes. Several have a
`--self-test` that runs without a GPU or a dataset.

### Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

The code reads two paths from the environment, both of which you set to your own locations:

```bash
export HD_REPO=/path/to/this/clone
export HD_DATA=/path/to/data          # generations and summaries are written here
```

`config.yaml` holds the models, the datasets and the generation settings. Every path inside the job
scripts is either derived from these variables or written as `/path/to/...` for you to replace.

### Running the pipeline

One model, one dataset, in order:

```bash
python 39_generate_dataset.py --model_folder <model> --dataset <dataset>
python 40_validate_dataset.py --model_folder <model> --dataset <dataset>
python 42_extract_phase2.py   --model_folder <model> --dataset <dataset>
python 44_eval_phase3.py      --model_folder <model> --dataset <dataset>
```

`44_eval_phase3.py` draws five splits, fits the Tucker-2 bases on training responses without labels,
trains the classifier, and reports pooled and within-question AUROC under both the question-level and
the answer-level split.

Baselines run through one harness, which owns the splits and the metrics so that no method computes
its own:

```bash
python 56_run_method.py --list
python 56_run_method.py --method <name> --model_folder <model> --dataset <dataset>
```

On a cluster, `submit_pipeline.sh <model>` queues the whole graph as SLURM jobs with the right
dependencies; the other `submit_*.sh` wrappers queue single stages and the ablations. Each wrapper
prints what it will submit before it submits anything, and `DRY=1` stops after printing.

### Reproducing the tables

| Result | Script |
|---|---|
| Main comparison, both splits | `44_eval_phase3.py`, `56_run_method.py`, `50_harp_reeval_prompt_level.py` |
| Within-question AUROC | `44_eval_phase3.py`, `50_harp_reeval_prompt_level.py --within` |
| Summary ablation | `44_eval_phase3.py` with the summary configurations |
| Rank sensitivity | `63_rank_sweep.py` |
| Layer window | `57_extract_all_layers.py`, `58_layer_sweep.py`, `64_window_sweep.py` |
| Reductions (flat PCA, random projection, layer average) | `61_flatten_control.py` |
| Tucker-3 with a token mode | `69_extract_token_states.py`, `70_tensor_tucker.py` |
| ACT-ViT | `59_extract_act_tensors.py`, `methods/act_vit.py`, `62_act_vit_joint.py` |
| Cross-model transfer | `65_extract_cross_read.py`, `66_cross_model.py` |
| Covariance spectra | `60_class_spectrum.py` |
| Label agreement with the released judge | `51_judge_agreement.py` |

### External code

HARP and ACT-ViT are run from their authors' released implementations, which are not redistributed
here. `49_harp_adapter.py` writes our generations into HARP's expected input format, and
`harp_patch_devices.sh` applies the only change we make to that code, which is to two device strings
so it runs on a single GPU. `59_extract_act_tensors.py` reproduces ACT-ViT's released preprocessing
exactly, and its self-test asserts bit-identity against their implementation when a clone is present.
