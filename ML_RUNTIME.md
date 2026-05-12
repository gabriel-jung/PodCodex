# ML runtime compatibility

Version pins, cache layout, and runtime patches for the ML stack. Single source of truth when a model load throws on bundle but works in dev, or vice versa.

## Pinned versions

| Package | Version | Why |
|---|---|---|
| `torch` | `2.8.0` (`+cu128` GPU, `+cpu` CPU) | Baseline for all ML. Drives cuda kernel matrix. |
| `transformers` | `==4.57.3` | Pinned exactly by `qwen-tts==0.1.1`. Cannot drift. |
| `qwen-tts` | `>=0.1.1` | Synth voice model wrapper. Owns the transformers pin above. |
| `pyannote.audio` | (range from `pyproject.toml`) | Diarization. Respects `HF_HOME` only. |
| `faster-whisper` | (range from `pyproject.toml`) | ASR. Takes explicit `download_root`. |
| `FlagEmbedding` (BGE-M3) | (range) | RAG embedder. Respects `HF_HOME` only. |

GPU wheel routing lives in `pyproject.toml [tool.uv.sources]`. CPU/Pascal/Turing+ extras are mutually exclusive; see `CLAUDE.md` for the lifecycle.

## HF model cache layout

All ML model files live under `<data_dir>/models/`:

```
<data_dir>/models/
  huggingface/
    hub/                            ← canonical HF snapshot store
      models--Qwen--Qwen3-TTS-...
      models--BAAI--bge-m3
      ...
  torch/                            ← torch.hub artifacts
  sentence-transformers/            ← ST cache
```

Three HF env vars **must all point at `<hf>/hub/`** or the loader/downloader halves split-brain:

| Env var | Used by | Required value |
|---|---|---|
| `HF_HOME` | pyannote, BGE-M3 (anything that doesn't take an explicit `cache_dir`) | `<data_dir>/models/huggingface/` |
| `HF_HUB_CACHE` | `huggingface_hub.snapshot_download` | `<data_dir>/models/huggingface/hub/` |
| `TRANSFORMERS_CACHE` | `transformers.from_pretrained`, qwen-tts internals | `<data_dir>/models/huggingface/hub/` |

**Split-brain symptom:** `OSError: Can't load feature extractor for <path>/preprocessor_config.json`. Cause: `snapshot_download` wrote to `hub/`, transformers read from `transformers/` (its fallback when `TRANSFORMERS_CACHE` is unset is `<HF_HOME>/transformers/`, a different dir). Both halves needed the snapshot, neither had a complete one.

### Setters (in precedence order)

1. **Tauri shell** (`src-tauri/src/lib.rs`, `set_env_for_sidecar`): bundled-app launch. Sets `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `SENTENCE_TRANSFORMERS_HOME`, `PODCODEX_DATA_DIR`. **Not** `HF_HOME` (would tell HF Hub to also look in `<HF_HOME>/hub/`, doubling the layout).
2. **Bundled sidecar entry** (`api/server.py:_wire_ml_caches`): runs from `main()`. `setdefault`s the same set, covering anything Tauri didn't pre-set. **Skipped** when `PODCODEX_DATA_DIR` is unset.
3. **Lazy Python** (`core/cache.py:get_hf_cache_dir`): called by every model loader before `from_pretrained`. `setdefault`s `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`. Covers the dev path (`make dev-api` runs `uvicorn` directly and never hits `_wire_ml_caches`).

All three are idempotent and converge. If any source diverges from `hub/`, fix it at the source; don't rebind in another setter.

## Transformers mask path bugs

`transformers==4.57.3 / masking_utils.py` has two opposing vmap bugs:

| Code path | Bug | Triggered by |
|---|---|---|
| `sdpa_mask_older_torch` | `NameError: TransformGetItemToIndex` (symbol gated by the same broken torch-version flag that selected this path) | Pplx's `or_masks` factory used by `rag/embedder.py` PplxEmbedder |
| `sdpa_mask_recent_torch` | `RuntimeError: vmap: ... .item() ...` on CPU | MiMi codec `packed_sequence_mask` (qwen-tts inside the synth subprocess) |

Dispatch is gated by `_is_torch_greater_or_equal_than_2_6`. In a PyInstaller bundle, `--copy-metadata torch` doesn't always expose torch's dist-info from `_MEIPASS`, so `importlib.metadata.version("torch")` raises and the gate misfires to `False` → older path → Pplx breaks.

### Workaround chain

Each subprocess hits `bootstrap_for_subprocess_child` → `_install_all_patches`:

1. **`_install_transformers_torch_check_patch`** (`bootstrap.py`): runs in every subprocess. Forces `_is_torch_greater_or_equal_than_2_6 = True`, rebinds `sdpa_mask = sdpa_mask_recent_torch`, injects `TransformGetItemToIndex` into `masking_utils` namespace. Fixes Pplx; exposes MiMi.
2. **`_patch_sdpa_mask_for_mimi_vmap_bug`** (`core/synthesize.py`): runs only inside the synth subprocess via `load_tts_model`. Replaces `_vmap_for_bhqkv` with a broadcast no-vmap implementation. Bypasses both vmap bugs simultaneously; all shipping mask functions (causal, padding, packed_sequence, sliding/chunked window, offsets, and/or composition) are pure tensor ops that broadcast cleanly.

Scoping the broadcast patch to the synth subprocess preserves the bootstrap recent-torch rebind for Pplx, pyannote, whisper, BGE-M3 (running in their own subprocesses). The broadcast approach would also work for them, but until verified across every caller we keep the scope narrow.

### Why we don't follow voicebox's approach

[jamiepine/voicebox](https://github.com/jamiepine/voicebox) source-patches `masking_utils.py` at PyInstaller import time to flip `_is_torch_greater_or_equal_than_2_6 = False`, forcing the older path. Works for them because they don't use Pplx. We do, and the older path crashes on `or_masks` regardless of our other fixes, so we can't accept it.

## Device routing

`core/device.py` is the single entry. Resolves to `"cuda"` or `"cpu"`; no MPS. `PODCODEX_DEVICE=auto|cpu|cuda` env override. `device.resolve_device()`, `device.cuda_available()`, `device.torch_dtype()`, `device.device_str()` are the canonical calls. Pascal (sm_60-62) needs `int8_float32` + `float32`; bfloat16 needs sm_80+. Bootstrap kernel guard sets `PODCODEX_DEVICE=cpu` when the wheel lacks kernels for the local GPU.

## Per-model notes

| Model | Cache mechanism | Notes |
|---|---|---|
| Qwen3-TTS 0.6B / 1.7B | `HF_HOME` + `TRANSFORMERS_CACHE` | CPU `float32`, CUDA `bfloat16` (when sm_80+). MiMi codec inside is the vmap-bug trigger. |
| BGE-M3 | `HF_HOME` only | `FlagEmbedding.BGEM3FlagModel` doesn't accept `cache_dir`. |
| Multilingual E5 small | `cache_folder=get_hf_cache_dir()` | Explicit. |
| Pplx embedder | `cache_dir=` + `cache_folder=` | Explicit. Hits transformers `or_masks` path. |
| WhisperX | `download_root=get_hf_cache_dir()` | Explicit. |
| Pyannote diarization | `HF_HOME` env | Needs `HF_TOKEN` for `pyannote/speaker-diarization-community-1`. Missing token causes silent hang at diarize. |

## When something breaks

1. **`OSError: ... preprocessor_config.json`** → HF cache split-brain. Check `TRANSFORMERS_CACHE` and `HF_HUB_CACHE` both point at `<hf>/hub/`. Inspect `<hf>/transformers/` for orphan snapshots; they're now dead weight, safe to delete.
2. **`RuntimeError: vmap ... .item()`** → MiMi path. Confirm `_patch_sdpa_mask_for_mimi_vmap_bug` is running in the synth subprocess (look for the function in `synthesize.py`; called from `load_tts_model`).
3. **`NameError: TransformGetItemToIndex`** → bootstrap torch-check patch didn't apply or transformers version drifted. Check `bootstrap.py:_install_transformers_torch_check_patch` logged its rebind.
4. **Pyannote hangs at diarize step** → missing `HF_TOKEN`.
5. **Silent CPU fallback after CUDA was expected** → kernel guard fired; check `bootstrap.py:_check_cuda_kernels_or_degrade` logs and `PODCODEX_DEVICE` env.

## When transformers / qwen-tts upgrades

Both vmap bugs may disappear upstream. If so, drop the patches in this order:

1. Confirm `transformers` version no longer pins `qwen-tts` exactly. If still pinned, the patch is still needed even if upstream fixed it (we can't move).
2. Remove `_patch_sdpa_mask_for_mimi_vmap_bug` from `synthesize.py` and the call site in `load_tts_model`.
3. Verify the bootstrap torch-check patch (`bootstrap.py:_install_transformers_torch_check_patch`) is still needed for the PyInstaller dist-info issue; that's a separate bug from the mask path. Likely still required.
