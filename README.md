# MiniMax-H3-NF4 FL2VA (simplified)

Minimal runner for [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) **MiniMax-H3-NF4** text-to-video+audio (FL2VA) with extreme disk offload.

The runner is intentionally small: `run_minimax_h3_nf4_fl2va.py` loads the four NF4 checkpoints, runs one smoke prompt, and writes `outputs/t2va.mp4`.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA and ~8GB+ VRAM
- ~8GB system RAM for the disk-offload path
- Large local disk for model caches; the first full cache is roughly 52GB

The script uses `torch.bfloat16`, `device="cuda"`, and DiffSynth's disk offload settings. CPU-only, MPS, and non-CUDA paths are not configured here.

## Setup

Install DiffSynth-Studio from source first:

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
python -m pip install -e DiffSynth-Studio
```

Then clone this repo and install its small dependency file:

```bash
git clone https://github.com/wxhior/minimax-h3-nf4-fl2va.git
cd minimax-h3-nf4-fl2va
python -m pip install -r requirements.txt
```

## Running

By default the script sets `DIFFSYNTH_SKIP_DOWNLOAD=True`, so it expects the MiniMax and DiffSynth weights to already be present in the ModelScope or Hugging Face caches. For a machine that still needs to download weights, override that default for the first run:

```bash
DIFFSYNTH_SKIP_DOWNLOAD=False python run_minimax_h3_nf4_fl2va.py
```

If the weights are already cached:

```bash
python run_minimax_h3_nf4_fl2va.py
```

The generated video is written to:

```text
outputs/t2va.mp4
```

## Runtime behavior

Current smoke settings are hard-coded in `run_minimax_h3_nf4_fl2va.py`:

| Setting | Value |
| --- | --- |
| Prompt | English speaking boy sample prompt |
| Resolution | `240x416` |
| Frames | `49` |
| Inference steps | `20` |
| Seed | `0` |
| FPS | `24` |
| Audio sample rate | `32000` |

To change prompt, resolution, frame count, steps, seed, or output filename, edit the constants and call arguments in `run_minimax_h3_nf4_fl2va.py`. There is no CLI argument parser in this simplified runner.

## Cache and disk notes

- Set `MODELSCOPE_CACHE` and/or `HF_HOME` before running if the default cache location is not on a large disk.
- `DIFFSYNTH_SKIP_DOWNLOAD=True` is useful for repeat runs on pre-populated machines, but it will fail on a cold machine without cached weights.
- The script sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless the variable is already present.
- The offload config keeps checkpoint staging on disk and only computes on CUDA with `torch.bfloat16`; slower disks will make startup and inference slower.

Example with explicit cache locations:

```bash
export MODELSCOPE_CACHE=/data/modelscope-cache
export HF_HOME=/data/hf-cache
DIFFSYNTH_SKIP_DOWNLOAD=False python run_minimax_h3_nf4_fl2va.py
```

## Troubleshooting

- **Missing weights on first run:** run with `DIFFSYNTH_SKIP_DOWNLOAD=False`, or pre-populate the ModelScope/Hugging Face cache.
- **CUDA out of memory:** keep the smoke settings (`240x416`, `49` frames, `20` steps) before increasing resolution, frames, or steps.
- **No CUDA device:** this runner is not configured for CPU/MPS; use a CUDA-capable NVIDIA environment.
- **Disk fills during download:** move `MODELSCOPE_CACHE` / `HF_HOME` to a larger volume before downloading.
