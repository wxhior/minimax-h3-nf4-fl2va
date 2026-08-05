# MiniMax-H3-NF4 FL2VA (simplified)

Minimal runner for [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) **MiniMax-H3-NF4** text-to-video+audio (FL2VA) with extreme disk offload (~8GB VRAM).

## Requirements

- NVIDIA GPU with ~8GB+ VRAM
- Plenty of disk for model weights (~52GB first download)
- Python 3.10+

## Setup

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

Then clone this repo and run:

```bash
git clone https://github.com/wxhior/minimax-h3-nf4-fl2va.git
cd minimax-h3-nf4-fl2va
python run_minimax_h3_nf4_fl2va.py
```

Output: `outputs/t2va.mp4`

## Notes

- First run downloads NF4 weights from ModelScope/HuggingFace.
- Smoke settings: `240x416`, `49` frames, `20` steps (safer on 8GB).
- Set `DIFFSYNTH_SKIP_DOWNLOAD=False` (or unset) if weights are not cached yet.
- Optionally point `MODELSCOPE_CACHE` / `HF_HOME` at a large disk.
