"""
Gradio UI for FineDance — generate dance videos from music.

Usage:
    conda activate FineNet
    python app.py
"""

import os
import tempfile

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr

# Monkey-patch gradio_client bug: additionalProperties can be a bool,
# but the code assumes it's always a dict.
import gradio_client.utils as _gc_utils

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type


def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)


_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
from generate_dance import load_model, generate, _setup_render_args
from render import MovieMaker

# Preload model once at startup
print("Loading model...")
MODEL = load_model()
print("Model loaded. Initializing renderer...")

# Create MovieMaker on the main thread so pyglet's signal handler works.
_setup_render_args()
VISUALIZER = MovieMaker(save_path=".")
print("Starting UI...")


def run(audio_path):
    if audio_path is None:
        raise gr.Error("Please upload a music file.")

    logs = []

    def log_fn(msg):
        logs.append(msg)
        print(msg)

    songname = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(tempfile.gettempdir(), f"{songname}_dance.mp4")

    generate(audio_path, output_path, model=MODEL, visualizer=VISUALIZER, log_fn=log_fn)

    return output_path, "\n".join(logs)


with gr.Blocks(title="FineDance") as demo:
    gr.Markdown("# FineDance\nUpload a music file to generate a 3D dance video.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Upload Music",
                type="filepath",
            )
            generate_btn = gr.Button("Generate Dance", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Generated Dance")
            status_output = gr.Textbox(label="Status", lines=6, interactive=False)

    generate_btn.click(
        fn=run,
        inputs=[audio_input],
        outputs=[video_output, status_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
