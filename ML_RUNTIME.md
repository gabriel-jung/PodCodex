# ML runtime compatibility

Version pins, cache layout, and runtime patches for the ML stack. Single source of truth when a model load throws on bundle but works in dev, or vice versa.

## Pinned versions

| Package | Version | Why |
|---|---|---|
| `torch` | `2.8.0` (`+cu128` GPU, `+cpu` CPU) | Baseline for all ML. Drives cuda kernel matrix. |
| `transformers` | `==4.57.3` | Pinned exactly by `qwen-tts==0.1.1`. Cannot drift. |
| `qwen-tts` | `==0.1.1` | Synth voice model wrapper. Owns the transformers pin above, so it is pinned exactly in the `pipeline` extra; `tests/test_mimi_mask_patch.py` checks the patched function still exists. |
| `pyannote.audio` | (range from `pyproject.toml`) | Diarization. Gets an explicit `cache_dir` from `diarize_file`. |
| `faster-whisper` | (range from `pyproject.toml`) | ASR. Takes explicit `download_root`. |
| `FlagEmbedding` (BGE-M3) | (range) | RAG embedder. Gets an explicit `cache_dir` from `BGEEmbedder`. |

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

`HF_HUB_CACHE` and `TRANSFORMERS_CACHE` **must both point at `<hf>/hub/`**, with `HF_HOME` at `<hf>/`, or the loader/downloader halves split-brain:

| Env var | Used by | Required value |
|---|---|---|
| `HF_HOME` | libraries that don't take an explicit cache dir | `<data_dir>/models/huggingface/` |
| `HF_HUB_CACHE` | `huggingface_hub.snapshot_download` | `<data_dir>/models/huggingface/hub/` |
| `TRANSFORMERS_CACHE` | `transformers.from_pretrained`, qwen-tts internals | `<data_dir>/models/huggingface/hub/` |

**Split-brain symptom:** `OSError: Can't load feature extractor for <path>/preprocessor_config.json`. Cause: `snapshot_download` wrote to `hub/`, transformers read from `transformers/` (its fallback when `TRANSFORMERS_CACHE` is unset is `<HF_HOME>/transformers/`, a different dir). Both halves needed the snapshot, neither had a complete one.

### Setters (in precedence order)

1. **Tauri shell** (`src-tauri/src/lib.rs`, `spawn_backend_if_needed`): bundled-app launch. Sets `PODCODEX_DATA_DIR`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `SENTENCE_TRANSFORMERS_HOME`, but not `HF_HOME`.
2. **`core/cache.py:wire_model_caches`**, called first by every `bootstrap_for_*()` (sidecar, MCP stdio, dev server, bot, step workers). `setdefault`s the whole set under `get_cache_dir()`, including `HF_HOME`, so values Tauri preset win and everything else is filled in. `PODCODEX_CACHE_DIR` moves every cache at once. Dev and the bot use the same tree as the app, so the in-app model list sees every download.

The setter has to run before anything imports `huggingface_hub`, which copies the cache vars into module constants on import. The eager bootstrap patches import transformers, so wiring later in a model loader is too late: pyannote and Qwen3-TTS weights silently land in `~/.cache/huggingface`, where the model list and delete never see them. `tests/test_cache_dirs.py` checks that every bootstrap entry point wires first. `get_hf_cache_dir` / `get_hf_hub_dir` only name the dirs. Pyannote and BGE-M3 get `get_hf_hub_dir()` explicitly, which also holds in a script with no bootstrap; WhisperX, E5 and Pplx have always taken `get_hf_cache_dir()` (the flat `<hf>/models--*` layout) and keep it so existing downloads stay valid. `list_cached_models` and `delete_cached_model` read both layouts. If a path diverges from `hub/`, fix it in `wire_model_caches`; don't rebind in another setter.

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

`core/device.py` is the single entry. Resolves to `"cuda"` or `"cpu"`; no MPS. `PODCODEX_DEVICE=auto|cpu|cuda` env override. `device.resolve_device()`, `device.cuda_available()`, `device.torch_dtype()`, `device.device_str()` are the canonical calls. Pascal (sm_60-62) needs `int8_float32` + `float32`; bfloat16 needs sm_80+. A GPU-panel choice is persisted as `device_override` in `<data_dir>/settings.json` and applied at bootstrap when the env var is unset. `cuda` raises only through `resolve_device()` (whisper, pyannote); `cuda_available()` just returns True. The kernel guard (`core/device.py:ensure_kernel_guard`) sets `PODCODEX_DEVICE=cpu` when the wheel lacks kernels for the local GPU; in the shipped sidecar it runs on the first device query, in dev and step workers at startup.

## Per-model notes

| Model | Cache mechanism | Notes |
|---|---|---|
| Qwen3-TTS 0.6B / 1.7B | `HF_HOME` + `TRANSFORMERS_CACHE` | CPU `float32`, CUDA `bfloat16` (when sm_80+). MiMi codec inside is the vmap-bug trigger. |
| BGE-M3 | `cache_dir=<hf>/hub` | Explicit. |
| Multilingual E5 small | `cache_folder=get_hf_cache_dir()` | Explicit. |
| Pplx embedder | `cache_dir=` + `cache_folder=` | Explicit. Hits transformers `or_masks` path. |
| WhisperX | `download_root=get_hf_cache_dir()` | Explicit. |
| Pyannote diarization | `cache_dir=<hf>/hub` | Needs `HF_TOKEN` for `pyannote/speaker-diarization-community-1`. A missing token fails fast with `HF_TOKEN not found`; an invalid token or unaccepted model terms surface as a Hub gated-repo error. |

## When something breaks

1. **`OSError: ... preprocessor_config.json`** → HF cache split-brain. Check `TRANSFORMERS_CACHE` and `HF_HUB_CACHE` both point at `<hf>/hub/`. Inspect `<hf>/transformers/` for orphan snapshots; they're now dead weight, safe to delete.
2. **`RuntimeError: vmap ... .item()`** → MiMi path. Confirm `_patch_sdpa_mask_for_mimi_vmap_bug` is running in the synth subprocess (look for the function in `synthesize.py`; called from `load_tts_model`).
3. **`NameError: TransformGetItemToIndex`** → bootstrap torch-check patch didn't apply or transformers version drifted. Check `bootstrap.py:_install_transformers_torch_check_patch` logged its rebind.
4. **`HF_TOKEN not found`, or a gated-repo error at diarize** → set `HF_TOKEN`, and accept the model terms on huggingface.co with the account that owns it.
5. **Models downloaded again, or missing from the in-app list** → something imported `huggingface_hub` before `wire_model_caches` ran; the weights are under `~/.cache/huggingface`. Check that the entry point calls a `bootstrap_for_*()` before any ML import.
6. **Silent CPU fallback after CUDA was expected** → kernel guard fired; check `core/device.py:ensure_kernel_guard` logs and `PODCODEX_DEVICE` env.

## When transformers / qwen-tts upgrades

Both vmap bugs may disappear upstream. If so, drop the patches in this order:

1. Confirm `transformers` version no longer pins `qwen-tts` exactly. If still pinned, the patch is still needed even if upstream fixed it (we can't move).
2. Remove `_patch_sdpa_mask_for_mimi_vmap_bug` from `synthesize.py` and the call site in `load_tts_model`.
3. Verify the bootstrap torch-check patch (`bootstrap.py:_install_transformers_torch_check_patch`) is still needed for the PyInstaller dist-info issue; that's a separate bug from the mask path. Likely still required.
