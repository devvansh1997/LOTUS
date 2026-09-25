# LLM Hallucination Detection via Low-rank Multilinear Decomposition

Code for the paper *LLM Hallucination Detection via Low-rank Multilinear Decomposition*, which
introduces LOTUS (low-rank order-statistic Tucker subspaces).

LOTUS summarizes the residual stream of a generated response by order statistics
over its tokens, compresses the resulting layer-by-feature matrices with a Tucker-2 decomposition whose
two bases are fitted to unlabeled training activations, and trains a classifier on the concatenated
cores. This repository contains the generation, extraction, evaluation and analysis code, plus the
cluster job scripts used to run it.

No data, model weights, logs or result files are included.

## Layout

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
```

Scripts are numbered by the order in which they entered the project, not by pipeline order; each file
begins with a docstring stating what it does, what it reads and what it writes. Several have a
`--self-test` that runs without a GPU or a dataset.

## Setup

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

## Running the pipeline

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

## Reproducing the tables

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

## External code

HARP and ACT-ViT are run from their authors' released implementations, which are not redistributed
here. `49_harp_adapter.py` writes our generations into HARP's expected input format, and
`harp_patch_devices.sh` applies the only change we make to that code, which is to two device strings
so it runs on a single GPU. `59_extract_act_tensors.py` reproduces ACT-ViT's released preprocessing
exactly, and its self-test asserts bit-identity against their implementation when a clone is present.
