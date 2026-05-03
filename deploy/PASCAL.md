# Pascal GPU support (GTX 10xx, Titan Xp, P40, P100)

The default PodCodex bundle ships PyTorch built against CUDA 12.8 (cu128).
Those wheels were compiled for **Turing (sm_75) and newer GPUs only** —
NVIDIA dropped Pascal kernels from the cu128 build matrix. If you have a
Pascal card, the bundle will start, the Tauri shell will appear, and the
first transcription run will crash inside torch with:

```
CUDA error: no kernel image is available for execution on the device
```

PodCodex's bootstrap kernel guard catches this on startup, sets
`PODCODEX_DEVICE=cpu`, and logs a warning — so the app stays usable in
CPU mode without manual intervention. To actually use the GPU, follow
one of the install paths below.

---

## Affected GPUs

| GPU | Architecture | Compute capability |
|-----|-------------|-------------------|
| GeForce GTX 1050 / 1060 / 1070 / 1080 (Ti) | Pascal GP102/104/106/107 | sm_61 |
| Titan X (Pascal) / Titan Xp | Pascal GP102 | sm_61 |
| Tesla P40 | Pascal GP102 | sm_61 |
| Tesla P100 | Pascal GP100 | sm_60 |
| Jetson TX2 / Tegra X2 | Pascal GP10B | sm_62 |

Anyone with Turing (RTX 20xx / GTX 16xx) or newer does not need this doc.

---

## Install path A — dev install via uv (recommended)

If you cloned the repo and run from source, add the `gpu-pascal` extra
instead of `gpu`:

```bash
git clone https://github.com/gabriel-jung/podcodex && cd podcodex
uv sync --extra desktop --extra pipeline --extra rag --extra youtube --extra mcp --extra gpu-pascal
```

`gpu-pascal` pulls torch + torchaudio from PyTorch's `cu126` index, which
still ships sm_60 + sm_61 kernels. Mutually exclusive with `gpu` — never
add both.

---

## Install path B — pre-built bundle + manual swap

If you installed the `.dmg` / `.msi` and don't want to clone the repo,
swap torch in the bundled venv after install. Bundled venv path per OS:

- **macOS:** N/A — Apple Silicon doesn't have CUDA. Pascal is x86 NVIDIA only.
- **Linux:** `~/.local/share/PodCodex/server-core/_internal/` (PyInstaller `--onedir` GPU sidecar).
- **Windows:** `%LOCALAPPDATA%\PodCodex\server-core\_internal\` (same layout).

```bash
# from the bundle's _internal directory
uv pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu126 \
    --reinstall --no-deps
```

---

## Verify it worked

```bash
uv run python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_name(0))"
```

Expected output includes `sm_60` (P100) or `sm_61` (everything else
listed above), and your GPU's name. If `sm_61` is missing, the install
didn't take — re-run with `--reinstall --force-reinstall`.

PodCodex's own diagnostic endpoint also reports it:

```bash
curl http://localhost:18811/api/system/device
```

```json
{
  "device": "cuda",
  "compute_type": "int8_float32",
  "torch_dtype": "float32",
  "compute_capability": "6.1",
  "gpu_name": "NVIDIA GeForce GTX 1080",
  "arch_list": ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_86"],
  "available": true,
  "override": null
}
```

`compute_type: int8_float32` is correct for Pascal — CTranslate2
(faster-whisper backend) refuses `float16` on Pascal because FP16 matmul
runs at 1/64 of FP32 speed there. INT8 matmul uses Pascal's DP4A
instructions (~native FP32 speed) and uses half the VRAM of FP32. For a
GTX 1080 (8 GB) running large-v3 + pyannote concurrently, this is the
sweet spot.

---

## CPU fallback

If the swap doesn't work, or you want to skip GPU entirely on this
machine, set the env var before launch:

```bash
PODCODEX_DEVICE=cpu          # Linux/macOS
$env:PODCODEX_DEVICE = "cpu" # Windows PowerShell
```

`make dev-no-tauri-cpu` does this for the dev flow. Bundle launchers
read `PODCODEX_DEVICE` from the parent shell's environment.

CPU mode uses CTranslate2's INT8 path. On modern x86 CPUs (Zen 3+,
Alder Lake+) Whisper large-v3 runs at ~3-4× real-time. Slower than a
GTX 1080 in INT8/FP32, but workable for one-off transcriptions.

---

## Long-term notes

- **PyTorch 2.10 is expected to drop Pascal entirely** (NVIDIA deprecated
  Pascal in CUDA 13). When PodCodex's torch baseline crosses 2.10, the
  `gpu-pascal` extra and this doc go away. Pin: the extra currently caps
  at `torch<2.10` so a stray `uv lock` doesn't silently upgrade you off
  the supported wheel.
- **NVIDIA drivers ≥ 535** are required for cu126 wheels. Older drivers
  on Pascal cards still get the same `no kernel image` error.
- **bfloat16 is not supported on Pascal** (needs sm_80). Qwen3-TTS
  voice synthesis on Pascal runs in float32 (~2× the VRAM of bfloat16,
  same quality). Confirmed working on a GTX 1080 with 8 GB.
