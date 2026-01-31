# [FineDance: A Fine-grained Choreography Dataset for 3D Full Body Dance Generation (ICCV 2023)](https://github.com/li-ronghui/FineDance)

[[Project Page](https://li-ronghui.github.io/finedance)] | [[Preprint](https://arxiv.org/abs/2212.03741)] | [[pdf](https://arxiv.org/pdf/2212.03741.pdf)] | [[video](https://li-ronghui.github.io/finedance)]

<img src="teaser/teaser.png">

## Quick Start

### Prerequisites

Install the conda environment and activate it:

```bash
conda env create -f environment.yaml
conda activate FineNet
```

Download the pretrained checkpoints and asset files from [Google Drive](https://drive.google.com/file/d/1ENoeUn-X-3Vw2Gon-voVLlndy3hZXdWD/view?usp=drive_link).

### Web UI (Recommended)

Launch the Gradio web interface:

```bash
python app.py
```

Open `http://127.0.0.1:7861` in your browser. Upload a music file and click "Generate Dance" to produce a video.

### Command Line

```bash
python generate_dance.py /path/to/music.mp3
```

Output will be saved to `output/<songname>_dance.mp4`.

To specify a custom output path:

```bash
python generate_dance.py /path/to/music.mp3 --output my_dance.mp4
```

Supported audio formats: any format ffmpeg can read (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, etc.).

### Output Details

- Resolution: 1200x1200
- Frame rate: 30 fps
- Duration: ~30 seconds
- Background: black
- Body model: SMPLX (full body with hands)

## How It Works

1. **Audio conversion** - Converts input to WAV if needed
2. **Feature extraction** - Slices audio into 4-second windows with 2-second stride, then extracts 35-dim features per slice (onset envelope, 20 MFCC, 12 chroma, peak onehot, beat onehot) using librosa
3. **Dance generation** - Feeds audio features into a pretrained diffusion model (`assets/checkpoints/train-2000.pt`) which generates SMPLX body motion (319-dim: 4 contact + 3 translation + 52 joints x 6 rotation)
4. **Rendering** - Converts generated motion to SMPLX meshes and renders 900 frames at 30fps using pyrender
5. **Final output** - Merges rendered video with original audio via ffmpeg

## FineDance Dataset

The dataset (7.7 hours) can be downloaded from [Google Drive](https://drive.google.com/file/d/1zQvWG9I0H4U3Zrm8d_QD_ehenZvqfQfS/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1gynUC7pMdpsE31wAwq177w?pwd=o9pw).

Put the downloaded data into `./data`. The data directory contains:

- **label_json** - Song name, coarse style, and fine-grained genre
- **motion** - [SMPLH](https://smpl-x.is.tue.mpg.de/) format motion data
- **music_wav** - Music data in WAV format
- **music_npy** - Music features extracted by [librosa](https://github.com/librosa/librosa) following [AIST++](https://github.com/google/aistplusplus_api/tree/main)

Reading a motion file:

```python
import numpy as np
data = np.load("motion/001.npy")
T, C = data.shape           # T is the number of frames
smpl_poses = data[:, 3:]
smpl_trans = data[:, :3]
```

### Dataset Split

The dataset is split into train, val, and test sets in two ways:

1. **FineDance@Genre** - Test set includes a broader range of dance genres; the same dancer may appear across splits but with different motions. Recommended for dance generation.
2. **FineDance@Dancer** - Splits are divided by dancer; the same dancer won't appear in different sets, but the test set contains fewer genres.

## Training

Only needed if you want to train from scratch. The pretrained checkpoint is already provided.

```bash
# Data preprocessing
python data/code/pre_motion.py

# Train
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

## Project Structure

```
FineDance/
├── app.py                   # Gradio web UI
├── generate_dance.py        # One-command dance generation (CLI)
├── train_seq.py             # Training script
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

## Acknowledgments

We would like to express our sincere gratitude to Dr [Yan Zhang](https://yz-cnsdqz.github.io/) and [Yulun Zhang](https://yulunzhang.com/) for their invaluable guidance and insights during the course of our research.

This code is based on: [EDGE](https://github.com/Stanford-TML/EDGE/tree/main), [MDM](https://github.com/Stanford-TML/EDGE/tree/main), [Adan](https://github.com/lucidrains/Adan-pytorch), [Diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch), [SMPLX](https://smpl-x.is.tue.mpg.de/).

## Citation

```
@inproceedings{li2023finedance,
  title={FineDance: A Fine-grained Choreography Dataset for 3D Full Body Dance Generation},
  author={Li, Ronghui and Zhao, Junfan and Zhang, Yachao and Su, Mingyang and Ren, Zeping and Zhang, Han and Tang, Yansong and Li, Xiu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10234--10243},
  year={2023}
}
```
