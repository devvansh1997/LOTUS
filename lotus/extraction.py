"""Hidden-state summaries of each response, read from a Hugging Face causal language model.

One forward pass per response over prompt + response, no generation. The hidden states of the
response's tokens at each block of the window are reduced to the five arrays in features.ARRAYS,
with the same pooling as the paper's code (57_extract_all_layers.pool_answer).
"""

import re
import time

import numpy as np

from ._version import __version__
from .features import ARRAYS, Features

# Blocks 15-23 (hidden-state indices 16-24): the window the paper used for all three of its models,
# which have 28, 32 and 44 blocks.
DEFAULT_BLOCKS = tuple(range(15, 24))
FLOAT16_MAX = 65504.0


def parse_blocks(text):
    """'15-23' -> (15, ..., 23); '10,12,14' -> (10, 12, 14). Blocks are 0-based."""
    text = str(text).strip()
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b < a:
            raise ValueError("layer range %s runs backwards" % text)
        blocks = tuple(range(a, b + 1))
    else:
        try:
            blocks = tuple(int(x) for x in text.split(","))
        except ValueError:
            raise ValueError("--layers takes a range such as 15-23 or a list such as 10,12,14; got %r" % text)
    if len(blocks) < 2:
        raise ValueError("the window needs at least two blocks (one summary is a difference between layers)")
    return blocks


def blocks_text(blocks):
    blocks = list(blocks)
    if blocks == list(range(blocks[0], blocks[-1] + 1)):
        return "%d-%d" % (blocks[0], blocks[-1])
    return ",".join(str(b) for b in blocks)


def summarize_tokens(h):
    """h: (P, T, D) hidden states of one response's T tokens at P blocks. Returns the five arrays."""
    a = np.asarray(h, dtype=np.float32)
    if a.ndim != 3 or a.shape[1] == 0:
        raise ValueError("expected (layers, tokens, channels) with at least one token; got %s" % (a.shape,))
    d = a[1:] - a[:-1]
    return {"core": a.max(axis=1),
            "q95": np.quantile(a, 0.95, axis=1).astype(np.float32),
            "q05": np.quantile(a, 0.05, axis=1).astype(np.float32),
            "v95": np.quantile(d, 0.95, axis=1).astype(np.float32),
            "v05": np.quantile(d, 0.05, axis=1).astype(np.float32)}


def text_config(config):
    """The part of a model config that describes the language model (multimodal models nest it)."""
    for c in (config, getattr(config, "text_config", None), getattr(config, "language_config", None)):
        if c is not None and getattr(c, "num_hidden_layers", None):
            return c
    raise ValueError("cannot find the number of layers (num_hidden_layers) in this model's config")


def _version_tuple(module):
    return tuple(int(x) for x in re.findall(r"\d+", module.__version__)[:2])


def resolve_device(device):
    import torch

    if device in (None, "", "default"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(name, device=None, dtype="auto", trust_remote_code=False):
    """(model, tokenizer) from a Hugging Face id or a local directory, in eval mode.

    device: cuda, cpu, cuda:1, ..., or auto to spread the model over the available GPUs (needs the
    accelerate package). dtype auto: bfloat16 on GPUs that support it, else float16; float32 on CPU."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(device)
    if dtype == "auto":
        if device == "cpu":
            dtype = "float32"
        else:
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    kwargs = {"trust_remote_code": trust_remote_code}
    kwargs["dtype" if _version_tuple(transformers) >= (4, 56) else "torch_dtype"] = torch_dtype
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    return model, tok


def token_ids(tokenizer, prompt, response, append_eos=True):
    """Prompt ids and response ids, tokenized separately so the boundary is exact, as the paper's code
    does when a model re-reads a response. The prompt gets the tokenizer's special tokens (such as BOS)
    unless it already starts with the BOS token, as a chat template's output does."""
    bos = getattr(tokenizer, "bos_token", None)
    add_special = not (bos and prompt.startswith(bos))
    p = list(tokenizer(prompt, add_special_tokens=add_special)["input_ids"])
    r = list(tokenizer(response, add_special_tokens=False)["input_ids"]) if response else []
    if append_eos:
        if tokenizer.eos_token_id is None:
            raise ValueError("the tokenizer has no end-of-sequence token; use append_eos=False (--no-eos)")
        r.append(int(tokenizer.eos_token_id))
    return p, r


def _rows(rows, limit=10):
    shown = ", ".join(str(i + 1) for i in rows[:limit])
    return shown + (", ..." if len(rows) > limit else "")


class _Silent:
    def update(self, n=1):
        pass

    def close(self):
        pass


def _progress(total, enabled):
    if not enabled:
        return _Silent()
    try:
        from tqdm import tqdm
    except ImportError:
        return _Silent()
    return tqdm(total=total, unit="batch", desc="reading")


def _allocate(n, n_layers, hidden_size, dtype):
    return {k: np.zeros((n, n_layers if k in ("core", "q95", "q05") else n_layers - 1, hidden_size),
                        dtype=dtype) for k in ARRAYS}


def extract(model, data, tokenizer=None, blocks=DEFAULT_BLOCKS, batch_size=4, device=None,
            dtype="auto", append_eos=True, trust_remote_code=False, store_dtype="float16",
            progress=True):
    """Summaries of every response in data (a Responses), as a Features.

    model       a Hugging Face id or local directory, or a loaded model (then pass tokenizer too)
    blocks      transformer blocks to read, 0-based; block b is hidden-state index b + 1
    append_eos  append the end-of-sequence token to each response, as in the paper, whose answer
                window runs through the stop token; it also gives an empty response a token to read
    store_dtype float16 (default) or float32; float16 switches to float32 by itself if a value
                exceeds its range
    """
    import torch

    if isinstance(model, str):
        name = model
        model, loaded_tok = load_model(model, device=device, dtype=dtype,
                                       trust_remote_code=trust_remote_code)
        tok = loaded_tok if tokenizer is None else tokenizer
    else:
        if tokenizer is None:
            raise ValueError("pass tokenizer= along with a loaded model")
        name = str(getattr(model.config, "_name_or_path", "") or type(model).__name__)
        tok = tokenizer
    if store_dtype not in ("float16", "float32"):
        raise ValueError("store_dtype must be float16 or float32")
    blocks = tuple(int(b) for b in blocks)
    if len(blocks) < 2:
        raise ValueError("the window needs at least two blocks")
    cfg = text_config(model.config)
    n_blocks = int(cfg.num_hidden_layers)
    if min(blocks) < 0 or max(blocks) >= n_blocks:
        raise ValueError("this model has %d blocks (0-%d), so the window %s does not fit; choose one "
                         "with --layers" % (n_blocks, n_blocks - 1, blocks_text(blocks)))
    if max(blocks) == n_blocks - 1:
        print("note: the window includes the model's last block, whose hidden state some models "
              "return after the final normalization")

    n = len(data)
    prompt_ids, response_ids = [], []
    for i in range(n):
        p, r = token_ids(tok, data.prompts[i], data.responses[i], append_eos)
        prompt_ids.append(p)
        response_ids.append(r)
    empty = [i for i in range(n) if not response_ids[i]]
    if empty:
        raise ValueError("%d responses have no tokens (rows %s); keep the end-of-sequence token "
                         "(the default) or drop them" % (len(empty), _rows(empty)))
    max_pos = getattr(cfg, "max_position_embeddings", None)
    if max_pos:
        too_long = [i for i in range(n) if len(prompt_ids[i]) + len(response_ids[i]) > max_pos]
        if too_long:
            raise ValueError("%d prompt + response pairs exceed the model's %d positions (rows %s)"
                             % (len(too_long), max_pos, _rows(too_long)))

    pad = tok.pad_token_id
    if pad is None:
        pad = tok.eos_token_id if tok.eos_token_id is not None else 0
    dev = getattr(model, "device", None) or next(model.parameters()).device
    order = sorted(range(n), key=lambda i: -(len(prompt_ids[i]) + len(response_ids[i])))
    starts = list(range(0, n, batch_size))
    store, widened = None, False
    bar = _progress(len(starts), progress)
    t0 = time.time()
    with torch.no_grad():
        for start in starts:
            idx = order[start:start + batch_size]
            seqs = [prompt_ids[i] + response_ids[i] for i in idx]
            width = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), width), int(pad), dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s, dtype=torch.long)
                mask[j, :len(s)] = 1
            out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False,
                        output_hidden_states=True)
            hidden = out.hidden_states
            for j, i in enumerate(idx):
                s, e = len(prompt_ids[i]), len(prompt_ids[i]) + len(response_ids[i])
                h = np.stack([hidden[b + 1][j, s:e].float().cpu().numpy() for b in blocks])
                summ = summarize_tokens(h)
                if store is None:
                    store = _allocate(n, len(blocks), h.shape[2], store_dtype)
                peak = max(float(np.abs(v).max()) for v in summ.values())
                if not np.isfinite(peak):
                    raise ValueError("row %d: the model returned non-finite hidden states; try "
                                     "--dtype float32" % (i + 1))
                if peak > FLOAT16_MAX and store["core"].dtype == np.float16:
                    store = {k: v.astype(np.float32) for k, v in store.items()}
                    widened = True
                for k in ARRAYS:
                    store[k][i] = summ[k]
            del out, hidden
            bar.update(1)
    bar.close()

    import transformers

    model_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    storage = str(store["core"].dtype)
    if widened:
        storage += " (a value exceeded the float16 range)"
    meta = {"model": name, "blocks": list(blocks), "model_blocks": n_blocks,
            "hidden_size": int(store["core"].shape[2]), "append_eos": bool(append_eos),
            "model_dtype": model_dtype, "storage": storage, "seconds": round(time.time() - t0, 1),
            "lotus_version": __version__, "transformers_version": transformers.__version__,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    return Features(store, data.labels, data.question_ids, data.ids,
                    n_tokens=[len(r) for r in response_ids], meta=meta)
