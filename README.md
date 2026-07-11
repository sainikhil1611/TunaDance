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

## Current blockers / asset provisioning

This project is containerized on a **best-effort basis**: the Docker image builds and all dependencies (including pytorch3d) are verified importable, but the actual dance-generation pipeline cannot run until you provide the external model assets below, and realistically needs a GPU. Read this section before filing an issue about a crash.

- **`pytorch3d` has no PyPI wheel and no CPU/macOS conda build.** `model/diffusion.py`, `render.py`, `vis.py`, and `dataset/quaternion.py` all `import pytorch3d.transforms` unconditionally — nothing that touches the model, training, or rendering pipeline can run without it. The `Dockerfile` builds it from source (`pip wheel --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable`) in a builder stage on Debian/Linux. **Verified outcome: this build SUCCEEDS** — pytorch3d 0.7.8 compiled against torch 2.13.0 on `python:3.12-slim-bookworm` (linux/arm64, ~4–6 min compile), and `pytorch3d.transforms` was confirmed importable and functionally working inside the built image. Two gotchas were required to make it work (both baked into the Dockerfile): `--no-build-isolation` (its setup.py imports torch at build time and fails immediately under pip's build isolation) and `MAX_JOBS=4` (unbounded parallel ninja jobs exhaust memory and kill the compile). If the build ever fails on your host, the image still builds — it ships without pytorch3d and the entry points fail with an explicit `ModuleNotFoundError` at import rather than something silent. To install it manually on a Linux machine or inside the devcontainer (which does **not** install pytorch3d automatically — only the production Dockerfile does):
  ```bash
  pip install torch setuptools ninja
  MAX_JOBS=4 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  ```
  macOS has no verified pytorch3d install path (upstream ships conda packages only for CUDA Linux; a from-source macOS build was not attempted here).
- **Model checkpoint + SMPLX body model are not in this repo.** `assets/` is gitignored and absent from git history entirely (external-asset redistribution isn't permitted). Download both from the [Google Drive link in the original release](https://drive.google.com/file/d/1ENoeUn-X-3Vw2Gon-voVLlndy3hZXdWD/view?usp=drive_link) and place them at:
  ```
  assets/checkpoints/train-2000.pt
  assets/smpl_model/smplx/SMPLX_NEUTRAL.npz
  ```
  Without these files, `app.py` now exits with a clear "Checkpoint not found" message instead of an unhandled traceback (fixed as part of containerization — see `app.py`'s `--checkpoint` flag and startup check). Mount them into the container at runtime, e.g. `docker run -v /path/to/assets:/app/assets ...`.
- **GPU caveat.** The container (and the Dockerfile's base image, `python:3.12-slim-bookworm`) is **CPU-only**. The code is written to be device-generic (`accelerate.Accelerator`, `torch.backends.mps.is_available()` checks), so it will *run* on CPU, but realistically both training (`train_seq.py`) and a full 900-frame inference render need a GPU (or Apple Silicon MPS, outside a Linux container) to complete in a reasonable time. Treat the container as good for verifying imports, `--help`, dependency installation, and the guarded failure paths — not for actually generating a dance video end to end.

## What's New (vs. upstream FineDance)

- **Gradio Web UI** (`app.py`) — Upload music in the browser, get a dance video back. No CLI knowledge required.
- **Single-command CLI** (`generate_dance.py`) — One command handles the full pipeline: audio feature extraction, diffusion sampling, SMPLX rendering, and audio-video muxing.
- **macOS / MPS support** — Updated `render.py`, `vis.py`, and inference code to run on Apple Silicon via MPS, with a dedicated `environment_macos.yaml`.
- **Accepts any audio format** — Automatically converts input to WAV via ffmpeg (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, etc.).
- **Finetuned checkpoint** — Finetuned the original FineDance model on additional data and for more epochs beyond the original 2000, improving dance quality and diversity.
- **Cleaned-up repo** — Removed wandb logs, debug scripts, and hardcoded paths.
- **Guarded entry points** — `app.py` checks for the checkpoint file and exits with a clear message instead of crashing; `generate_dance.py` exits with a clear message if `ffmpeg` isn't installed, instead of an uncaught `CalledProcessError`.

## Model Details

| | |
|---|---|
| **Architecture** | Transformer decoder with Gaussian diffusion |
| **Input** | 35-dim audio features (onset, 20 MFCC, 12 chroma, peak/beat onehot) per 4s window |
| **Output** | SMPLX body motion — 319-dim (4 contact + 3 translation + 52 joints x 6 rotation) |
| **Checkpoint** | `assets/checkpoints/train-2000.pt` (finetuned beyond 2000 epochs on additional data) |
| **Body model** | SMPLX (full body with hands) |
| **Training data** | FineDance dataset (7.7 hours of music-dance pairs) |

## Quick start (devcontainer — recommended)

1. Prereqs: [Docker](https://docs.docker.com/get-docker/), [VS Code](https://code.visualstudio.com/) + the "Dev Containers" extension.
2. Open this folder in VS Code → "Reopen in Container" when prompted. This installs `ffmpeg`, rendering system libs, and the pinned `requirements.txt` (pytorch3d is **not** installed by the devcontainer's `postCreateCommand` — install it in the container with the one-liner from "Current blockers" above; the production `Dockerfile` builds it automatically).
3. Download the checkpoint + SMPLX model (see "Current blockers / asset provisioning") and place them under `assets/`.
4. Run: `python app.py` → open http://127.0.0.1:7860 (Gradio's default port; the original upstream docs mention 7861, but the code does not set a port explicitly, so it uses Gradio's default of 7860).

## Manual setup (no devcontainer)

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements.txt
# pytorch3d is not in requirements.txt — install separately (see "Current blockers" above)
```

Install `ffmpeg` at the OS level (Debian/Ubuntu: `apt-get install ffmpeg`; macOS: `brew install ffmpeg`).

Download the pretrained checkpoint and SMPLX model from [Google Drive](https://drive.google.com/file/d/1ENoeUn-X-3Vw2Gon-voVLlndy3hZXdWD/view?usp=drive_link) and place them under `assets/` as described above.

### Web UI

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser. Upload a music file and click **Generate Dance**.

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

Only needed to train from scratch. The pretrained checkpoint is included via the Google Drive download above.

```bash
python data/code/pre_motion.py                              # preprocess
accelerate launch train_seq.py --batch_size 32 --epochs 200 # train
```

Key flags:
- `--batch_size` — Default 400; reduce to 32 or lower for Mac MPS / CPU
- `--epochs` — Default 2000
- `--checkpoint` — Resume from a saved checkpoint

Training requires the FineDance dataset (7.7 hours), available from [Google Drive](https://drive.google.com/file/d/1zQvWG9I0H4U3Zrm8d_QD_ehenZvqfQfS/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1gynUC7pMdpsE31wAwq177w?pwd=o9pw), placed under `./data`. This is a training-only requirement, not needed for inference.

## Environment variables

None. No API keys, no `.env` file — the only external requirements are the checkpoint/SMPLX files and `ffmpeg`, both handled above.

## Testing

No pytest-discoverable test suite exists in this repo (`test.py` is the original FineDance batch-inference/eval script, not a unit test suite — it requires the checkpoint and a music directory to run). What was actually verified, all inside the built `tunadance` production image unless noted:

- `python -m compileall .` — passes (pure syntax check).
- `python -c "from pytorch3d.transforms import axis_angle_to_quaternion; ..."` — pytorch3d 0.7.8 imports and computes correctly.
- `python app.py --help` — exits 0 with usage text, without loading gradio/torch/the model (verified after the `--help`-before-heavy-imports fix; also verified on the host with no deps installed at all, and inside the devcontainer).
- `python app.py` with no checkpoint present — exits 1 with a clear "Checkpoint not found" message, no traceback.
- `python generate_dance.py --help` — exits 0 with usage text. This exercises the full module import chain (`train_seq` → `model/diffusion` → `pytorch3d.transforms`, `render` → `pyrender`), so it confirms every Python dependency imports in the container.
- `python generate_dance.py fake.mp3` with ffmpeg removed from the container — exits with a clear "ffmpeg not found on PATH" message, no traceback.
- Devcontainer — `devcontainer up` succeeds (installs ffmpeg + `requirements.txt`), and `app.py --help` exits 0 inside it.
- Actual dance generation (`generate()` / the Gradio callback / `test.py`) was **not** run — it requires the checkpoint and SMPLX model, which are external Google Drive downloads not available in this environment.

## Deployment

Build the production image: `docker build -t tunadance .`
Run it: `docker run -v /path/to/assets:/app/assets -p 7860:7860 tunadance`
This container runs on any host (Fly.io, Render, a VPS with Docker) — see
[DEPLOYMENT.md](../DEPLOYMENT.md) in the Projects root for the full walkthrough.

## Known limitations

- **pytorch3d install is solved only inside the production Docker image.** No PyPI wheel exists; the Dockerfile's from-source Linux build works (verified, see "Current blockers" above), and the same recipe should work on any Linux box — but the devcontainer does not install it automatically, and macOS has no known-working pytorch3d path at all as of this writing.
- **External assets are not included and cannot be fetched automatically** — the checkpoint, SMPLX model, and full training dataset all live behind Google Drive/Baidu Cloud links that this environment cannot download.
- **GPU strongly recommended for real use.** The container is CPU-only; actual generation (diffusion sampling + 900-frame SMPLX rendering) is expected to be very slow on CPU even once the assets are in place.
- **Generation was not end-to-end verified** in this containerization pass — only imports (including pytorch3d), `--help` behavior, and the guarded failure paths (missing checkpoint, missing ffmpeg) were confirmed. Everything past model loading is unverified without the external assets.
