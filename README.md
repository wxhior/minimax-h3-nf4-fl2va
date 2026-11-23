# MiniMax-H3-NF4 FL2VA (simplified)

Minimal runner for [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
**MiniMax-H3-NF4** text-to-video+audio (FL2VA) with extreme disk offload for
small-GPU smoke tests.

## What this runner does

`run_minimax_h3_nf4_fl2va.py` is intentionally a single-file example. It:

1. Creates `outputs/` next to the script.
2. Sets conservative defaults:
   - `DIFFSYNTH_SKIP_DOWNLOAD=True`
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
3. Loads the MiniMax-H3 NF4 FL2VA, text encoder, video VAE, audio VAE, and
   processor configs through DiffSynth.
4. Runs one text-to-video+audio prompt at `240x416`, `49` frames, `20`
   inference steps, and seed `0`.
5. Writes `outputs/t2va.mp4` at 24 FPS with 32 kHz audio.

## Requirements

- NVIDIA GPU with CUDA and roughly 8 GB+ VRAM.
- Enough local disk for model weights; the first uncached download is about
  52 GB.
- Python 3.10+.
- DiffSynth-Studio installed from source.

## Setup

Install DiffSynth-Studio first:

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

Then clone and run this repo:

```bash
git clone https://github.com/wxhior/minimax-h3-nf4-fl2va.git
cd minimax-h3-nf4-fl2va
python run_minimax_h3_nf4_fl2va.py
```

Output: `outputs/t2va.mp4`

## First run and model caches

The script defaults `DIFFSYNTH_SKIP_DOWNLOAD` to `True`, so an uncached first
run will not download missing weights unless you override it:

```bash
DIFFSYNTH_SKIP_DOWNLOAD=False python run_minimax_h3_nf4_fl2va.py
```

Use a large disk for model caches when the default cache locations are small:

```bash
MODELSCOPE_CACHE=/data/modelscope-cache \
HF_HOME=/data/hf-cache \
DIFFSYNTH_SKIP_DOWNLOAD=False \
python run_minimax_h3_nf4_fl2va.py
```

After weights are cached, you can omit the override and keep the script's
download-skip default.

## Customizing a smoke run

Edit the constants near the bottom of `run_minimax_h3_nf4_fl2va.py`:

- `prompt`: text and spoken audio content to generate.
- `height`, `width`, `num_frames`, `num_inference_steps`: workload size.
- `seed`: reproducibility control.
- `out_path`: output file path under `outputs/`.

The checked-in settings are deliberately small for 8 GB-class GPUs. Raising
resolution, frames, or steps increases runtime and memory pressure.

## Troubleshooting

- **Missing model files**: set `DIFFSYNTH_SKIP_DOWNLOAD=False` for the run, or
  confirm that `MODELSCOPE_CACHE` / `HF_HOME` point to caches that already
  contain the MiniMax-H3-NF4 weights.
- **CUDA out of memory**: keep the default `240x416`, `49` frame, `20` step
  smoke settings first. Increase only one dimension at a time.
- **Cache or disk errors**: move `MODELSCOPE_CACHE` and `HF_HOME` to a disk
  with enough free space before running.
- **CUDA unavailable**: this script hard-codes `computation_device="cuda"`;
  run it on a CUDA-capable NVIDIA machine.
