# FineDance: Quick Start Guide

## Prerequisites

Activate the conda environment before running any commands:

```bash
conda activate FineNet
```

## Web UI (Recommended)

Launch the Gradio web interface:

```bash
python app.py
```

Then open `http://127.0.0.1:7861` in your browser. Upload a music file and click "Generate Dance" to produce a video.

## Command Line

```bash
python generate_dance.py /path/to/music.mp3
```

Output will be saved to `output/<songname>_dance.mp4`.

To specify a custom output path:

```bash
python generate_dance.py /path/to/music.mp3 --output my_dance.mp4
```

### Supported audio formats

Any format ffmpeg can read: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, etc.

### What happens under the hood

1. **Audio conversion** - Converts input to WAV if needed
2. **Feature extraction** - Slices audio into 4-second windows with 2-second stride, then extracts 35-dim features per slice (onset envelope, 20 MFCC, 12 chroma, peak onehot, beat onehot) using librosa
3. **Dance generation** - Feeds audio features into a pretrained diffusion model (`assets/checkpoints/train-2000.pt`) which generates SMPLX body motion (319-dim: 4 contact + 3 translation + 52 joints x 6 rotation)
4. **Rendering** - Converts generated motion to SMPLX meshes and renders 900 frames at 30fps using pyrender
5. **Final output** - Merges rendered video with original audio via ffmpeg

### Output details

- Resolution: 1200x1200
- Frame rate: 30 fps
- Duration: ~30 seconds
- Background: black
- Body model: SMPLX (full body with hands)

## Project Structure

```
FineDance/
├── app.py                   # Gradio web UI
├── generate_dance.py        # One-command dance generation (CLI)
├── train_seq.py             # Training script (not needed if using pretrained checkpoint)
├── test.py                  # Original test/inference script
├── render.py                # Video rendering (SMPLX mesh to MP4)
├── args.py                  # CLI argument definitions
├── vis.py                   # Skeleton/FK utilities
├── assets/
│   ├── checkpoints/
│   │   └── train-2000.pt    # Pretrained model (2000 epochs)
│   └── smpl_model/
│       └── smplx/
│           └── SMPLX_NEUTRAL.npz  # SMPLX body model
├── model/
│   ├── model.py             # SeqModel (transformer decoder)
│   └── diffusion.py         # Gaussian diffusion (training + sampling)
├── dataset/
│   └── FineDance_dataset.py # Dataset loader
└── data/
    └── finedance/           # Training data (music + motion pairs)
```

## Training (Optional)

Only needed if you want to train from scratch. The pretrained checkpoint is already provided.

```bash
accelerate launch train_seq.py --batch_size 32 --epochs 200
```

Key flags:
- `--batch_size` - Default is 400, reduce to 32 or lower for Mac MPS (limited to ~30GB)
- `--epochs` - Default is 2000
- `--checkpoint` - Resume from a saved checkpoint

## Advanced Usage

### Generate on the test set

```bash
python data/code/slice_music_motion.py
python generate_all.py --motion_save_dir generated/finedance_seq_120_dancer --save_motions
```

### Render a pre-generated motion file

```bash
python render.py --modir eval/motions --mode smplx
```
