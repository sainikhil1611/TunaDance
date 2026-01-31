"""
End-to-end dance generation from a music file.

Usage:
    python generate_dance.py /path/to/music.mp3
    python generate_dance.py /path/to/music.mp3 --output my_dance.mp4
"""

import argparse
import glob
import os
import subprocess
import sys
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import librosa
import librosa as lr
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from train_seq import EDGE
from render import MovieMaker, motion_data_load_process


# --- Audio utilities (from test.py) ---

def slice_audio(audio_file, stride, length, out_dir):
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx


def extract_features(fpath, full_seq_len=120):
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH

    data, _ = librosa.load(fpath, sr=SR)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)
    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T

    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
    )
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0

    start_bpm = lr.beat.tempo(y=lr.load(fpath)[0])[0]
    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
        start_bpm=start_bpm, tightness=100,
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0

    audio_feature = np.concatenate(
        [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
        axis=-1,
    )
    audio_feature = audio_feature[:4 * FPS]
    return audio_feature


key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])

def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0

stringintkey = cmp_to_key(stringintcmp_)


# --- Model loading ---

class _Opt:
    """Minimal config namespace for EDGE model."""
    feature_type = "baseline"
    full_seq_len = 120
    windows = 10
    nfeats = 319
    do_normalize = False
    datasplit = "cross_genre"
    project = "experiments/finedance_seq_120_genre/train"
    exp_name = "finedance_seq_120_genre"
    render_dir = "tmp_renders"
    batch_size = 64
    epochs = 1
    save_interval = 10
    ema_interval = 1
    checkpoint = ""
    wandb_pj_name = "finedance_seq"


def load_model(checkpoint_path="assets/checkpoints/train-2000.pt"):
    """Load the EDGE model once. Returns (model, opt)."""
    opt = _Opt()
    model = EDGE(opt, opt.feature_type, checkpoint_path)
    model.eval()
    return model, opt


def _setup_render_args():
    """Inject render.py global args for MovieMaker."""
    import render as render_module
    render_module.args = argparse.Namespace(
        mode="smplx", fps=30, gpu="0", modir="", save_path=None
    )


# --- Main pipeline ---

def generate(music_path, output_path, model=None, visualizer=None, log_fn=print):
    """
    Generate a dance video from a music file.

    Args:
        music_path: Path to input audio (mp3, wav, etc.)
        output_path: Where to save the output mp4
        model: Pre-loaded (model, opt) tuple. If None, loads fresh.
        visualizer: Pre-built MovieMaker instance. If None, creates one.
        log_fn: Callable for status messages (default: print)
    """
    import shutil

    music_path = os.path.abspath(music_path)
    songname = os.path.splitext(os.path.basename(music_path))[0]

    # Step 1: Convert to WAV if needed
    log_fn(f"[1/5] Preparing audio: {os.path.basename(music_path)}")
    temp_root = TemporaryDirectory()
    wav_dir = os.path.join(temp_root.name, "wav")
    os.makedirs(wav_dir)
    wav_path = os.path.join(wav_dir, songname + ".wav")

    if music_path.lower().endswith(".wav"):
        shutil.copy2(music_path, wav_path)
    else:
        subprocess.run(
            ["ffmpeg", "-i", music_path, wav_path, "-y"],
            capture_output=True, check=True,
        )

    # Step 2: Slice and extract features
    log_fn("[2/5] Extracting audio features...")
    slice_dir = os.path.join(temp_root.name, "slices")
    os.makedirs(slice_dir)
    stride = 60 / 30  # 2 seconds
    full_seq_len = 120
    slice_audio(wav_path, stride, full_seq_len / 30, slice_dir)

    file_list = sorted(glob.glob(f"{slice_dir}/*.wav"), key=stringintkey)
    out_length = 30  # seconds
    sample_size = int(out_length / stride) - 1

    cond_list = []
    for file in tqdm(file_list[:sample_size]):
        reps = extract_features(file)[:full_seq_len]
        cond_list.append(reps)
    cond = torch.from_numpy(np.array(cond_list))
    filenames = file_list[:sample_size]

    # Step 3: Generate motion
    log_fn("[3/5] Generating dance motion...")
    if model is None:
        edge_model, opt = load_model()
    else:
        edge_model, opt = model

    motion_dir = os.path.join(temp_root.name, "motions")
    os.makedirs(motion_dir)

    data_tuple = (None, cond, filenames)
    edge_model.render_sample(
        data_tuple, "gen", temp_root.name, render_count=-1,
        fk_out=motion_dir, mode="long", render=False,
    )

    # Step 4: Render video
    log_fn("[4/5] Rendering video...")
    motion_file = glob.glob(os.path.join(motion_dir, "*.pkl"))[0]
    modata = motion_data_load_process(motion_file)

    video_dir = os.path.join(temp_root.name, "video")
    os.makedirs(video_dir)

    _setup_render_args()
    if visualizer is None:
        visualizer = MovieMaker(save_path=video_dir)
    else:
        visualizer.save_path = video_dir
    visualizer.run(modata, tab=songname, music_file=wav_path)

    # Step 5: Copy final output
    log_fn("[5/5] Saving output...")
    rendered_file = os.path.join(video_dir, songname + "z.mp4")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.move(rendered_file, output_path)

    temp_root.cleanup()
    log_fn(f"Done! Output saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dance video from a music file.")
    parser.add_argument("music", type=str, help="Path to the input music file (mp3, wav, etc.)")
    parser.add_argument("--output", type=str, default=None, help="Output video path (default: output/<songname>_dance.mp4)")
    args = parser.parse_args()

    if args.output is None:
        songname = os.path.splitext(os.path.basename(args.music))[0]
        args.output = os.path.join("output", f"{songname}_dance.mp4")

    generate(args.music, args.output)
