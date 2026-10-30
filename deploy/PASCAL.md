# Pascal GPU support (GTX 10xx, Titan Xp, P40, P100)

PodCodex's GPU builds use PyTorch for CUDA 12.8, which has no kernels for Pascal cards (compute capability sm_60-62: GTX 1050-1080 Ti, Titan X/Xp, Tesla P40 and P100, Jetson TX2). On those cards the kernel guard (`core/device.py`) detects the missing kernels, switches PodCodex to CPU and logs a warning, so the app keeps working without the GPU.

Turing (RTX 20xx / GTX 16xx) and newer don't need this page.

## Using the GPU: run from source

The installed app can't use a Pascal GPU: its in-app GPU backend is a frozen build of the CUDA 12.8 torch, and nothing in it can be swapped. Run from source with the `gpu-pascal` extra instead, which pulls torch from PyTorch's `cu126` index (still built with Pascal kernels):

```bash
git clone https://github.com/gabriel-jung/PodCodex && cd PodCodex
uv sync --extra desktop --extra pipeline --extra rag --extra youtube --extra mcp --extra bot --extra gpu-pascal
make setup-frontend       # npm install
make dev-no-tauri         # or `make dev` for the native window (needs Rust)
```

Never run `make setup` or `make setup-python` on this checkout: they re-sync without `gpu-pascal`, which swaps the cu126 torch back out and silently puts you on the CPU. `gpu-pascal`, `gpu` and `cpu` are mutually exclusive. cu126 wheels need NVIDIA driver 535 or newer.

To force the CPU anyway, use the GPU panel in Settings or set `PODCODEX_DEVICE=cpu` (`$env:PODCODEX_DEVICE = "cpu"` in PowerShell).

Check that it took:

```bash
uv run python -c "import torch; print(torch.cuda.get_arch_list(), torch.cuda.get_device_name(0))"
```

The list must include `sm_60` (P100) or `sm_61` (the others). If not, re-run the `uv sync` above with `--reinstall`. With the app running, Settings → GPU acceleration should then show the card in use.

## What to expect

- Transcription runs with `compute_type=int8_float32`: CTranslate2 refuses `float16` on Pascal, where FP16 is very slow, and INT8 runs at about FP32 speed in half the VRAM. A GTX 1080 (8 GB) fits large-v3 plus pyannote.
- Voice synthesis runs in `float32`, since bfloat16 needs sm_80 (about twice the VRAM, same quality; works on a GTX 1080).
- On CPU instead (the installed app, or `PODCODEX_DEVICE=cpu`), Whisper large-v3 runs at about 3-4× real time on a recent x86 CPU.

## Lifetime

PyTorch 2.10 is expected to drop Pascal entirely. The `gpu-pascal` extra caps torch below 2.10, and it and this page go away when PodCodex's torch baseline moves past that.
