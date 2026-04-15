# Multi-Angle Batch Renderer

Batch-render all camera angle variations of a single image using Qwen Image Edit models on [Comfy Cloud](https://cloud.comfy.org).

When training LoRAs or building 3D assets, you often need systematic multi-angle renders of a subject. Manually running ComfyUI workflows for every azimuth, elevation, and distance combination is tedious. This script automates the entire process — submit one image, get back a complete set of angle variations.

## Supported Pipelines

| Pipeline | Angles LoRA | Unique Poses | Camera Control |
|----------|-------------|-------------|----------------|
| **2511** (Nov 2025) | [fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) | 96 (8 az x 4 el x 3 dist) | `QwenMultiangleCameraNode` with numeric angles |
| **2509** (Sep 2025) | [dx8152/Qwen-Edit-2509-Multiple-angles](https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles) | 72 (8 az x 3 el x 3 dist) | Bilingual text prompts |

### Angle Grid

- **Azimuths** (8): 0, 45, 90, 135, 180, 225, 270, 315 degrees
- **Elevations** (4 for 2511, 3 for 2509): -30, 0, 30, 60 degrees
- **Distances** (3): 0.6 (close-up), 1.0 (medium), 1.8 (wide)

## Setup

```bash
pip install aiohttp
export COMFY_CLOUD_API_KEY="your-key-here"  # from https://cloud.comfy.org
```

The required models and LoRAs must be available in your Comfy Cloud workspace.

## Usage

```bash
# 2511 pipeline (default, 96 poses)
python batch_multi_angle.py --image photo.png --cloud

# 2509 pipeline (72 poses)
python batch_multi_angle.py --image photo.png --cloud --pipeline 2509

# Different seed (output dir auto-named by pipeline + seed)
python batch_multi_angle.py --image photo.png --cloud --seed 123

# Append text to every prompt
python batch_multi_angle.py --image photo.png --cloud --prompt-append "dramatic lighting"

# Render a subset
python batch_multi_angle.py --image photo.png --cloud --azimuths 0,90,180,270 --elevations 0

# Preview all prompts without rendering
python batch_multi_angle.py --image photo.png --cloud --dry-run
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | (required) | Input image path |
| `--cloud` | off | Use Comfy Cloud (otherwise targets local ComfyUI) |
| `--pipeline` | `2511` | `2509` or `2511` |
| `--output` | auto | Output directory (default: `./multi_angle_output_{pipeline}_seed{seed}`) |
| `--seed` | `42` | Random seed |
| `--steps` | `4` | Inference steps (Lightning LoRA tuned for 4) |
| `--guidance` | `1.0` | CFG scale |
| `--concurrency` | `3` | Parallel cloud jobs |
| `--lora-angles` | `1.0` | Angles LoRA strength |
| `--lora-lightning` | `1.0` | Lightning LoRA strength |
| `--azimuths` | all | Subset, e.g. `0,90,180,270` |
| `--elevations` | all | Subset, e.g. `-30,0,30` |
| `--distances` | all | Subset, e.g. `0.6,1.0` |
| `--prompt-append` | `""` | Text appended to every prompt |
| `--timeout` | `600` | Per-job timeout in seconds |
| `--dry-run` | off | Print prompts without rendering |

## Output

Images are saved with descriptive filenames:

```
az000_el+00_d1.0_front_view_eyelevel_shot_medium_shot.png
az090_el-30_d0.6_right_side_view_lowangle_shot_closeup.png
az180_el+30_d1.8_back_view_elevated_shot_wide_shot.png
```

Existing files are automatically skipped, so you can safely re-run to fill in any gaps.

## How It Works

1. Uploads the input image to Comfy Cloud
2. Connects a WebSocket to receive real-time execution results
3. Submits all angle variation workflows with concurrency control
4. Downloads completed renders as they finish via the WebSocket output events
5. Skips any poses that already have output files

## License

MIT
