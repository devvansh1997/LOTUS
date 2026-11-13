"""Tests for the lotus package. Run: python tests/test_lotus.py (pytest also collects them).

Paper equivalence: the tests load this repository's original scripts (26, 28, 43, 44, 57) and check
that the package reproduces their scaling, Tucker-2 bases, cores, splits, metrics, token pooling and
end-to-end AUROC exactly, on synthetic data.

Extraction: a small randomly initialised 28-block model and its tokenizer are built in a temporary
directory, so nothing is downloaded and no GPU is needed. A GPU, if present, is exercised too.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import lotus  # noqa: E402
from lotus import Features, Lotus, Responses  # noqa: E402
from lotus.detector import apply_robust_scale, fit_robust_scale, project_core, tucker2_bases  # noqa: E402
from lotus.extraction import summarize_tokens, token_ids  # noqa: E402

_PAPER, _TMP, _TINY = {}, [], {}


def paper(filename):
    """One of the repository's numbered scripts, loaded as a module (its main() does not run)."""
    if filename not in _PAPER:
        path = os.path.join(ROOT, filename)
        if not os.path.exists(path):
            raise unittest.SkipTest("%s is not in this checkout" % filename)
        spec = importlib.util.spec_from_file_location("paper_" + filename.split("_")[0], path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except ImportError as e:
            raise unittest.SkipTest("cannot import %s: %s" % (filename, e))
        _PAPER[filename] = mod
    return _PAPER[filename]


def tmpdir(prefix):
    d = tempfile.mkdtemp(prefix=prefix)
    _TMP.append(d)
    return d


def synthetic(n_questions=60, per_question=10, n_layers=9, hidden=96, seed=0):
    """Stored arrays (float16, as extraction writes them), labels and question ids, with a planted
    signal. Six questions are never answered correctly, so the paper protocol has unknown questions."""
    rng = np.random.default_rng(seed)
    qids = np.repeat(np.arange(n_questions), per_question)
    p_wrong = rng.uniform(0.05, 0.95, n_questions)
    p_wrong[:6] = 1.0
    y = (rng.random(len(qids)) < p_wrong[qids]).astype(np.int64)
    direction = rng.standard_normal(hidden)

    def block(n_l):
        x = rng.standard_normal((len(qids), n_l, hidden))
        x += 0.6 * y[:, None, None] * direction * rng.uniform(0.5, 1.5, (1, n_l, 1))
        return x.astype(np.float16)

    arrays = {"core": block(n_layers), "q95": block(n_layers), "q05": block(n_layers),
              "v95": block(n_layers - 1), "v05": block(n_layers - 1)}
    return arrays, y, qids


# --------------------------------------------------------------------------------------------------
# Equivalence with the paper's code
# --------------------------------------------------------------------------------------------------

def test_robust_scaling_matches_paper():
    m28 = paper("28_eval_band.py")
    X = (np.random.default_rng(1).standard_normal((300, 50)) * 3).astype(np.float32)
    X[:5] *= 40
    ours, theirs = fit_robust_scale(X[:200]), m28.fit_robust_scale(X[:200])
    for k in theirs:
        assert np.array_equal(ours[k], theirs[k]), k
    assert np.array_equal(apply_robust_scale(X, ours), m28.apply_robust_scale(X, theirs))


def test_tucker_bases_and_cores_match_paper():
    m43 = paper("43_eval_phase2.py")
    X = np.random.default_rng(2).standard_normal((120, 9, 80)).astype(np.float32)
    for seed in (0, 7):
        U_L, U_D = tucker2_bases(X, 5, 64, seed)
        P_L, P_D = m43.compute_ul_ud_randomized(X, 5, 64, seed)
        assert np.array_equal(U_L, P_L) and np.array_equal(U_D, P_D)
        assert np.array_equal(project_core(X, U_L, U_D), m43.s01.project_core(X, P_L, P_D))
    # scaling fitted on training rows, bases from the scaled training rows, cores for every row
    train = np.arange(0, 120, 2)
    params = fit_robust_scale(X[train].reshape(len(train), -1))
    Xs = apply_robust_scale(X.reshape(len(X), -1), params).reshape(X.shape)
    core = project_core(Xs, *tucker2_bases(Xs[train], 5, 64, 3))
    assert np.array_equal(core, m43.fold_pure_core_randomized(X, train, 5, 64, 3))


def test_paper_splits_match_paper():
    m26, m43 = paper("26_grouped_baseline.py"), paper("43_eval_phase2.py")
    _, y, qids = synthetic()
    is_known, _ = m43.derive_is_known(y, qids)
    for seed, (train, test) in lotus.paper_splits(qids, y):
        t, v = m26.original_harp_split(is_known, qids, len(y), seed=seed)
        assert np.array_equal(train, t) and np.array_equal(test, v), seed
        assert not set(qids[train]) & set(qids[test])


def test_question_splits_keep_questions_whole():
    _, _, qids = synthetic()
    n_q = len(np.unique(qids))
    first = lotus.question_splits(qids)
    for seed, (train, test) in first:
        assert not set(qids[train]) & set(qids[test])
        assert len(train) + len(test) == len(qids)
        assert len(np.unique(qids[train])) == int(n_q * 0.75)
    again = lotus.question_splits(qids)
    assert all(np.array_equal(a[1][1], b[1][1]) for a, b in zip(first, again))
    assert not np.array_equal(first[0][1][1], first[1][1][1])


def test_within_question_auroc_matches_paper():
    m26 = paper("26_grouped_baseline.py")
    _, y, qids = synthetic()
    scores = np.round(np.random.default_rng(3).random(len(y)), 2)
    ours = lotus.within_question_auroc(scores, y, qids)
    theirs = m26.within_prompt_auroc(scores, y, qids)
    assert ours["auroc"] == theirs["within_prompt_auroc"]
    assert ours["n_mixed_questions"] == theirs["n_mixed_prompts"]
    assert ours["n_pairs"] == theirs["n_pairs"]


def test_token_pooling_matches_paper():
    m57 = paper("57_extract_all_layers.py")
    h = np.random.default_rng(4).standard_normal((9, 13, 32)).astype(np.float32)
    ours = summarize_tokens(h)
    theirs = m57.pool_answer(h, ("core", "static", "velocity"))
    for k in lotus.ARRAYS:
        assert np.array_equal(ours[k], theirs[k]), k
    one = summarize_tokens(h[:, :1])
    assert np.array_equal(one["core"], h[:, 0]) and np.array_equal(one["q95"], h[:, 0])


def test_end_to_end_matches_paper_pipeline():
    """Same features, splits and seeds: the package's cores and per-seed AUROC equal the paper's."""
    from sklearn.metrics import roc_auc_score

    m44 = paper("44_eval_phase3.py")
    arrays, y, qids = synthetic()
    feats = Features(arrays, y, qids)
    seeds = (42, 0, 1)
    splits = lotus.paper_splits(qids, y, seeds=seeds)
    ours = lotus.evaluate_lotus(feats, splits)["per_seed"]
    built = m44.build_feats(arrays["core"], arrays["q95"], arrays["q05"], arrays["v95"], arrays["v05"])
    builder = m44.make_grouped_builder(m44.CONDITION_SPECS["triple_concat"], built)
    is_known, _ = m44.derive_is_known(y, qids)
    for k, seed in enumerate(seeds):
        t, v = m44.harp_split(is_known, qids, len(y), seed)
        assert np.array_equal(t, splits[k][1][0]) and np.array_equal(v, splits[k][1][1])
        Z = builder(t, seed)
        det = Lotus(seed=seed).fit_projection(feats.take(t))
        assert np.array_equal(det.transform(feats), Z), "cores differ for seed %d" % seed
        auc = 100.0 * roc_auc_score(y[v], m44.fit_eval("RF", Z[t], y[t], Z[v], seed))
        assert ours[k]["pooled_auroc"] == auc, (seed, ours[k]["pooled_auroc"], auc)
        assert auc > 60, "the planted signal should be detectable (got %.1f)" % auc


def test_phase2_reader_and_check_script():
    from sklearn.metrics import roc_auc_score

    m44 = paper("44_eval_phase3.py")
    arrays, y, qids = synthetic(seed=5)
    path = os.path.join(tmpdir("lotus_phase2_"), "toy_phase2_features.npz")
    np.savez(path, static_max=arrays["core"], static_q95=arrays["q95"], static_q05=arrays["q05"],
             velocity_q95=arrays["v95"], velocity_q05=arrays["v05"], label=y, prompt_id=qids)
    feats = Features.from_phase2_npz(path)
    assert np.array_equal(feats.labels, y) and np.array_equal(feats.question_ids, qids)
    built = m44.build_feats(arrays["core"], arrays["q95"], arrays["q05"], arrays["v95"], arrays["v05"])
    builder = m44.make_grouped_builder(m44.CONDITION_SPECS["triple_concat"], built)
    is_known, _ = m44.derive_is_known(y, qids)
    aucs = []
    for seed in (42, 0):
        t, v = m44.harp_split(is_known, qids, len(y), seed)
        Z = builder(t, seed)
        aucs.append(100.0 * roc_auc_score(y[v], m44.fit_eval("RF", Z[t], y[t], Z[v], seed)))
    res = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "check_against_paper.py"),
                          "--features", path, "--seeds", "42", "0", "--expect", "%.2f" % np.mean(aucs),
                          "--paper-code"], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "identical" in res.stdout and "match" in res.stdout


# --------------------------------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------------------------------

def test_responses_validation():
    r = Responses.from_records([{"question_id": 3, "prompt": "p", "response": "r", "label": 1},
                                {"question_id": 12, "prompt": "p", "response": "", "label": 0}])
    assert r.question_ids.dtype.kind == "i" and r.labels.tolist() == [1, 0] and r.has_labels
    bad_rows = [({"question_id": 1, "prompt": "p", "response": "r", "label": 2}, "label must be 0"),
                ({"question_id": 1, "prompt": "p"}, "missing field"),
                ({"question_id": 1, "prompt": 5, "response": "r"}, "must be a string")]
    for bad, msg in bad_rows:
        try:
            Responses.from_records([bad])
        except ValueError as e:
            assert msg in str(e), str(e)
        else:
            raise AssertionError("accepted %r" % bad)
    path = os.path.join(tmpdir("lotus_jsonl_"), "rows.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"question_id": "a", "prompt": "p", "response": "r", "label": 0}\n{not json}\n')
    try:
        Responses.from_jsonl(path)
    except ValueError as e:
        assert "line 2" in str(e), str(e)
    else:
        raise AssertionError("accepted a line that is not JSON")


def test_features_and_detector_roundtrip():
    arrays, y, qids = synthetic(n_questions=30, seed=6)
    feats = Features(arrays, y, qids, meta={"blocks": list(range(15, 24)), "model": "toy"})
    d = tmpdir("lotus_roundtrip_")
    feats.save(os.path.join(d, "f.npz"))
    back = Features.load(os.path.join(d, "f.npz"))
    for k in lotus.ARRAYS:
        assert np.array_equal(back.arrays[k], feats.arrays[k]) and back.arrays[k].dtype == np.float16
    assert np.array_equal(back.labels, y) and np.array_equal(back.question_ids, qids)
    assert back.meta == feats.meta
    det = Lotus(seed=3).fit(feats.take(np.arange(200)))
    assert det.transform(feats).shape == (len(feats), 896) == (len(feats), det.n_features)
    det.save(os.path.join(d, "det.joblib"))
    again = Lotus.load(os.path.join(d, "det.joblib"))
    assert np.array_equal(det.score(feats), again.score(feats))
    try:
        Lotus().fit(feats.take(np.where(y == 0)[0]))
    except ValueError as e:
        assert "both 0" in str(e)
    else:
        raise AssertionError("trained on one class")


# --------------------------------------------------------------------------------------------------
# Extraction, on a small randomly initialised model
# --------------------------------------------------------------------------------------------------

def tiny_model_dir():
    """A 28-block LLaMA-style model (hidden size 64) and a byte-level BPE tokenizer that adds BOS,
    saved to a temporary directory. Built once per run, from scratch: nothing is downloaded."""
    if "dir" in _TINY:
        return _TINY["dir"]
    try:
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    except ImportError as e:
        raise unittest.SkipTest("needs torch, tokenizers and transformers: %s" % e)
    corpus = ["Q: What is the capital of France?\nA: Paris is the capital of France.",
              "Q: Who wrote Hamlet?\nA: William Shakespeare wrote it around 1600.",
              "Q: Which vitamin deficiency causes scurvy?\nA: Vitamin C, also called ascorbic acid.",
              "Q: How many bones are in the adult human body?\nA: There are 206 bones.",
              "Q: question number 7?\nA: answer 3!! and more text, digits 0123456789."]
    bpe = Tokenizer(models.BPE(unk_token="<unk>"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, show_progress=False,
                                  special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    bpe.train_from_iterator(corpus * 20, trainer)
    bos = bpe.token_to_id("<s>")
    bpe.post_processor = processors.TemplateProcessing(single="<s> $A", pair="<s> $A <s> $B",
                                                       special_tokens=[("<s>", bos)])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, bos_token="<s>", eos_token="</s>",
                                  unk_token="<unk>", pad_token="<pad>")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=28,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                      pad_token_id=tok.pad_token_id)
    model = LlamaForCausalLM(cfg)
    d = tmpdir("lotus_tiny_model_")
    model.save_pretrained(d)
    tok.save_pretrained(d)
    _TINY["dir"] = d
    return d


def tiny_rows():
    rows = [("q1", "Q: What is the capital of France?\nA:", " Paris", 0),
            ("q1", "Q: What is the capital of France?\nA:", " Lyon is the capital of France.", 1),
            ("q2", "Q: Who wrote Hamlet?\nA:", "", 1),
            ("q3", "Q: How many bones are in the adult human body?\nA:",
             " There are 206 bones in the adult human body, and more at birth.", 0),
            ("q3", "Q: How many bones are in the adult human body?\nA:", " 300", 1)]
    return Responses(*zip(*rows))


def test_tokenization_boundaries():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir())
    p, r = token_ids(tok, "Q: What is two plus two?\nA:", " four")
    assert p[0] == tok.bos_token_id and p.count(tok.bos_token_id) == 1
    assert r[-1] == tok.eos_token_id and tok.decode(r[:-1]) == " four"
    p2, _ = token_ids(tok, tok.bos_token + "Q: hi\nA:", " x")
    assert p2.count(tok.bos_token_id) == 1, "a prompt that already starts with BOS must not get a second"
    assert token_ids(tok, "Q: hi\nA:", "", append_eos=True)[1] == [tok.eos_token_id]
    assert token_ids(tok, "Q: hi\nA:", "", append_eos=False)[1] == []
    try:  # a real tokenizer, only if it is already in the local cache
        real = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", local_files_only=True)
    except Exception:
        return
    prompt, resp = "Q: Which vitamin deficiency causes scurvy?\nA:", " Vitamin C, also called ascorbic acid."
    p, r = token_ids(real, prompt, resp)
    assert real.decode(p + r[:-1]) == prompt + resp and r[-1] == real.eos_token_id


def test_extract_matches_single_forward():
    """Batched, padded extraction equals one forward pass per response pooled by the paper's code."""
    import torch

    m57 = paper("57_extract_all_layers.py")
    model, tok = lotus.load_model(tiny_model_dir(), device="cpu", dtype="float32")
    data = tiny_rows()
    feats = lotus.extract(model, data, tokenizer=tok, batch_size=3, store_dtype="float32", progress=False)
    assert feats.meta["blocks"] == list(range(15, 24)) and feats.meta["model_blocks"] == 28
    for i in range(len(data)):
        p, r = token_ids(tok, data.prompts[i], data.responses[i])
        assert feats.n_tokens[i] == len(r)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([p + r]), output_hidden_states=True)
        h = np.stack([out.hidden_states[b + 1][0, len(p):].float().numpy() for b in lotus.DEFAULT_BLOCKS])
        ref = m57.pool_answer(h, ("core", "static", "velocity"))
        for k in lotus.ARRAYS:
            np.testing.assert_allclose(feats.arrays[k][i], ref[k], rtol=1e-4, atol=1e-5, err_msg=k)
    half = lotus.extract(model, data, tokenizer=tok, batch_size=3, progress=False)
    for k in lotus.ARRAYS:
        assert half.arrays[k].dtype == np.float16
        assert np.array_equal(half.arrays[k], feats.arrays[k].astype(np.float16)), k
    try:
        lotus.extract(model, data, tokenizer=tok, blocks=range(20, 30), progress=False)
    except ValueError as e:
        assert "does not fit" in str(e)
    else:
        raise AssertionError("accepted a window past the last block")
    try:
        lotus.extract(model, data, tokenizer=tok, append_eos=False, progress=False)
    except ValueError as e:
        assert "no tokens" in str(e)
    else:
        raise AssertionError("accepted an empty response without the end-of-sequence token")


def test_extract_on_gpu():
    import torch

    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA device")
    d = tiny_model_dir()
    gpu_model, tok = lotus.load_model(d, device="cuda")
    cpu_model, _ = lotus.load_model(d, device="cpu", dtype="float32")
    data = tiny_rows()
    gpu = lotus.extract(gpu_model, data, tokenizer=tok, batch_size=2, progress=False)
    cpu = lotus.extract(cpu_model, data, tokenizer=tok, batch_size=2, store_dtype="float32", progress=False)
    assert gpu.meta["model_dtype"] in ("bfloat16", "float16")
    for k in lotus.ARRAYS:
        a, b = gpu.arrays[k].astype(np.float32).ravel(), cpu.arrays[k].ravel()
        assert np.corrcoef(a, b)[0, 1] > 0.99, k


def test_cli_end_to_end():
    d = tiny_model_dir()
    work = tmpdir("lotus_cli_")
    rows = [{"question_id": "q%02d" % q, "prompt": "Q: question number %d?\nA:" % q,
             "response": " answer %d%s" % (k, "!" * (k % 3)), "label": int((q + k) % 3 == 0),
             "id": "r%d_%d" % (q, k)} for q in range(12) for k in range(6)]
    data_path = os.path.join(work, "rows.jsonl")
    with open(data_path, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")

    def run(*args, code=0):
        res = subprocess.run([sys.executable, "-m", "lotus"] + list(args), cwd=ROOT,
                             capture_output=True, text=True)
        assert res.returncode == code, "lotus %s exited %d:\n%s\n%s" % (" ".join(args), res.returncode,
                                                                       res.stdout, res.stderr)
        return res

    feats_path = os.path.join(work, "f.npz")
    out = run("extract", "--model", d, "--data", data_path, "--out", feats_path, "--device", "cpu",
              "--batch-size", "8").stdout
    assert "blocks 15-23" in out and "43 x 64 values per response" in out, out
    out = run("evaluate", "--features", feats_path, "--seeds", "0", "1").stdout
    assert "pooled AUROC (%)" in out and "within-question (%)" in out, out
    with open(os.path.join(work, "f.lotus.json"), encoding="utf-8") as f:
        lotus_doc = json.load(f)
    first = lotus_doc["per_seed"][0]
    assert lotus_doc["seeds"] == [0, 1] and len(first["test_scores"]) == first["n_test"]
    assert first["test_ids"][0].startswith("r")
    run("evaluate", "--features", feats_path, "--protocol", "paper", "--seeds", "0",
        "--out", os.path.join(work, "paper.json"))
    splits_path = os.path.join(work, "splits.json")
    run("splits", "--data", data_path, "--out", splits_path, "--seeds", "0", "1")
    scores_path = os.path.join(work, "other.txt")
    np.savetxt(scores_path, np.random.default_rng(5).random(len(rows)))
    run("evaluate", "--scores", scores_path, "--data", data_path, "--splits", splits_path)
    with open(os.path.join(work, "other.eval.json"), encoding="utf-8") as f:
        other_doc = json.load(f)
    assert [s["test_rows"] for s in other_doc["per_seed"]] == [s["test_rows"] for s in lotus_doc["per_seed"]]
    det_path = os.path.join(work, "det.joblib")
    run("fit", "--features", feats_path, "--out", det_path)
    run("score", "--detector", det_path, "--features", feats_path, "--out", os.path.join(work, "s.txt"))
    assert len(np.loadtxt(os.path.join(work, "s.txt"))) == len(rows)
    err = run("evaluate", "--features", feats_path, "--scores", scores_path, code=2).stderr
    assert err.startswith("error:"), err


def _cleanup():
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)


def main():
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    start = time.time()
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
        except unittest.SkipTest as e:
            skipped += 1
            print("SKIP  %-42s %s" % (name, e), flush=True)
            continue
        except Exception:
            failed += 1
            print("FAIL  %s" % name, flush=True)
            traceback.print_exc()
            continue
        print("PASS  %-42s %5.1f s" % (name, time.time() - t0), flush=True)
    print("\n%d passed, %d failed, %d skipped in %.0f s"
          % (len(tests) - failed - skipped, failed, skipped, time.time() - start))
    _cleanup()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
