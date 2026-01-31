---
license: mit
tags:
  - dance-generation
  - motion-synthesis
  - diffusion
  - smplx
  - music-to-dance
  - 3d-motion
library_name: pytorch
pipeline_tag: audio-to-video
base_model: li-ronghui/FineDance
---

# TunaDance

**Music-to-dance generation with a Gradio web UI and single-command CLI.**

TunaDance builds on [FineDance](https://github.com/li-ronghui/FineDance) (ICCV 2023), a diffusion-based model that generates full-body 3D dance from music. This fork finetunes the original model on additional data and for more epochs beyond the original 2000, and adds a user-friendly interface layer and macOS support so you can go from an audio file to a rendered dance video without touching the inference internals.

[[Original Paper](https://arxiv.org/abs/2212.03741)] | [[Original Project Page](https://li-ronghui.github.io/finedance)] | [[Original Repo](https://github.com/li-ronghui/FineDance)]

<img src="teaser/teaser.png">

## What's New (vs. upstream FineDance)

- **Gradio Web UI** (`app.py`) — Upload music in the browser, get a dance video back. No CLI knowledge required.
- **Single-command CLI** (`generate_dance.py`) — One command handles the full pipeline: audio feature extraction, diffusion sampling, SMPLX rendering, and audio-video muxing.
- **macOS / MPS support** — Updated `render.py`, `vis.py`, and inference code to run on Apple Silicon via MPS, with a dedicated `environment_macos.yaml`.
- **Accepts any audio format** — Automatically converts input to WAV via ffmpeg (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, etc.).
- **Finetuned checkpoint** — Finetuned the original FineDance model on additional data and for more epochs beyond the original 2000, improving dance quality and diversity.
- **Cleaned-up repo** — Removed wandb logs, debug scripts, and hardcoded paths.

## Model Details

| | |
|---|---|
| **Architecture** | Transformer decoder with Gaussian diffusion |
| **Input** | 35-dim audio features (onset, 20 MFCC, 12 chroma, peak/beat onehot) per 4s window |
| **Output** | SMPLX body motion — 319-dim (4 contact + 3 translation + 52 joints x 6 rotation) |
| **Checkpoint** | `assets/checkpoints/train-2000.pt` (finetuned beyond 2000 epochs on additional data) |
| **Body model** | SMPLX (full body with hands) |
| **Training data** | FineDance dataset (7.7 hours of music-dance pairs) |

## Quick Start

### Prerequisites

```bash
# Install conda environment
conda env create -f environment.yaml        # Linux/CUDA
conda env create -f environment_macos.yaml   # macOS (Apple Silicon)

conda activate FineNet
```

Download the pretrained checkpoint and SMPLX model from [Google Drive](https://drive.google.com/file/d/1ENoeUn-X-3Vw2Gon-voVLlndy3hZXdWD/view?usp=drive_link) and place them under `assets/`.

### Web UI (Recommended)

```bash
python app.py
```

Open `http://127.0.0.1:7861` in your browser. Upload a music file and click **Generate Dance**.

### Command Line

```bash
python generate_dance.py /path/to/music.mp3
```

Output is saved to `output/<songname>_dance.mp4`. Use `--output` for a custom path:

```bash
python generate_dance.py /path/to/music.mp3 --output my_dance.mp4
```

### Output Specs

| Property | Value |
|---|---|
| Resolution | 1200 x 1200 |
| Frame rate | 30 fps |
| Duration | ~30 seconds |
| Body model | SMPLX (full body with hands) |

## How It Works

1. **Audio conversion** — Converts input to WAV if needed via ffmpeg
2. **Feature extraction** — Slices audio into 4s windows (2s stride), extracts 35-dim features using librosa
3. **Dance generation** — Diffusion model generates SMPLX motion sequence from audio features
4. **Rendering** — Converts motion to SMPLX meshes, renders 900 frames at 30fps with pyrender
5. **Muxing** — Merges rendered video with original audio via ffmpeg

## Training

Only needed to train from scratch. The pretrained checkpoint is included.

```bash
python data/code/pre_motion.py                              # preprocess
accelerate launch train_seq.py --batch_size 32 --epochs 200 # train
```

Key flags:
- `--batch_size` — Default 400; reduce to 32 or lower for Mac MPS
- `--epochs` — Default 2000
- `--checkpoint` — Resume from a saved checkpoint

## FineDance Dataset

The dataset (7.7 hours) is available from [Google Drive](https://drive.google.com/file/d/1zQvWG9I0H4U3Zrm8d_QD_ehenZvqfQfS/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1gynUC7pMdpsE31wAwq177w?pwd=o9pw). Place it under `./data`.

```python
import numpy as np
data = np.load("motion/001.npy")
smpl_poses = data[:, 3:]   # joint rotations
smpl_trans = data[:, :3]   # root translation
```

Two dataset splits are provided:
- **FineDance@Genre** (recommended) — Broader genre coverage in the test set
- **FineDance@Dancer** — Splits by dancer identity

## Project Structure

```
TunaDance/
├── app.py                   # Gradio web UI  [NEW]
├── generate_dance.py        # End-to-end CLI [NEW]
├── environment_macos.yaml   # macOS conda env [NEW]
├── train_seq.py             # Training script
├── test.py                  # Original inference script
├── render.py                # SMPLX mesh rendering (updated for MPS)
├── vis.py                   # Skeleton/FK utilities (updated for MPS)
├── args.py                  # CLI argument definitions
├── assets/
│   ├── checkpoints/
│   │   └── train-2000.pt    # Pretrained model (2000 epochs)
│   └── smpl_model/
│       └── smplx/
│           └── SMPLX_NEUTRAL.npz
├── model/
│   ├── model.py             # SeqModel (transformer decoder)
│   └── diffusion.py         # Gaussian diffusion
├── dataset/
│   └── FineDance_dataset.py
└── data/
    └── finedance/           # Training data (music + motion pairs)
```

## Acknowledgments

This project is built on [FineDance](https://github.com/li-ronghui/FineDance) by Li et al. We thank the original authors for their work.

Upstream acknowledgments: [EDGE](https://github.com/Stanford-TML/EDGE/tree/main), [MDM](https://github.com/Stanford-TML/EDGE/tree/main), [Adan](https://github.com/lucidrains/Adan-pytorch), [Diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch), [SMPLX](https://smpl-x.is.tue.mpg.de/).

## Citation

```bibtex
@inproceedings{li2023finedance,
  title={FineDance: A Fine-grained Choreography Dataset for 3D Full Body Dance Generation},
  author={Li, Ronghui and Zhao, Junfan and Zhang, Yachao and Su, Mingyang and Ren, Zeping and Zhang, Han and Tang, Yansong and Li, Xiu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10234--10243},
  year={2023}
}
```
