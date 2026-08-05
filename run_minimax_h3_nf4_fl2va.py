"""MiniMax-H3-NF4 FL2VA — 8GB VRAM extreme disk offload."""
import os
from pathlib import Path

import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: keep ModelScope / HF caches on a large disk
# os.environ.setdefault("MODELSCOPE_CACHE", "/path/to/modelscope-cache")
# os.environ.setdefault("HF_HOME", "/path/to/hf-cache")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Extreme hardware optimization: disk → VRAM, needs ~8GB VRAM / ~8GB system RAM
vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": "disk",
    "preparing_device": "disk",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

print("Loading MiniMax-H3-NF4 (will download ~52GB on first run)...")
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-fl2va-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="audio_vae_nf4.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=0,
)

prompt = "A boy is very happy, he is speaking in english: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
print("Running T2VA inference (240x416, 49 frames, 20 steps)...")
video, audio = pipe(
    prompt=prompt,
    height=240,
    width=416,
    num_frames=49,
    num_inference_steps=20,
    seed=0,
)
out_path = OUT_DIR / "t2va.mp4"
write_video_audio(
    video=video,
    audio=audio,
    output_path=str(out_path),
    fps=24,
    audio_sample_rate=32000,
)
print(f"Done: {out_path}")
