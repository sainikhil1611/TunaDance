# TunaDance — best-effort container image.
#
# pytorch3d has no PyPI wheel and no CPU-friendly conda build for this
# platform. The `pytorch3d-builder` stage below attempts a from-source
# build on Linux (this often works for CPU-only ops, unlike macOS). If
# that build fails, the final image ships WITHOUT pytorch3d and the
# entry points fail with a clear, guarded error message instead of a
# crash — see README.md "Current blockers / asset provisioning" for the
# manual install recipe and the actual verified outcome of this build.
FROM python:3.12-slim-bookworm AS pytorch3d-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.13.0 setuptools ninja

# Best-effort: build pytorch3d's CPU extension from source. Two gotchas,
# both hit and fixed during verification of this Dockerfile:
#   1. Its setup.py imports torch, so build isolation must be disabled
#      (with isolation the build env has no torch and fails immediately
#      with ModuleNotFoundError: No module named 'torch').
#   2. Unbounded parallel ninja jobs exhaust memory and the compile dies
#      with "Error compiling objects for extension"; MAX_JOBS=4 fixes it.
# With those two fixes the CPU build was verified to succeed on
# linux/arm64 (pytorch3d-0.7.8, torch 2.13.0, ~6 min compile). It is
# still allowed to fail without aborting the image build (marker files
# below, checked in the final stage) since it's unofficial upstream.
ENV MAX_JOBS=4
RUN mkdir -p /wheels && \
    ( pip wheel --no-cache-dir --no-deps --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
        -w /wheels \
      && touch /wheels/BUILD_OK \
      || touch /wheels/BUILD_FAILED )

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libegl1 libgles2 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pytorch3d from the wheel built above only if that build
# succeeded; otherwise leave it out and let the guarded entry points
# explain what's missing (model/diffusion.py, render.py, vis.py,
# dataset/quaternion.py all import pytorch3d.transforms).
COPY --from=pytorch3d-builder /wheels /wheels
RUN if [ -f /wheels/BUILD_OK ]; then \
        pip install --no-cache-dir /wheels/pytorch3d*.whl; \
    else \
        echo "pytorch3d source build failed during image build — shipping without it." >&2; \
    fi \
    && rm -rf /wheels

COPY . .

# Model checkpoint + SMPLX assets are not redistributable in this image
# (external Google Drive download, gitignored). Mount them at runtime:
#   docker run -v /path/to/assets:/app/assets ...
# See README.md for the download steps.

# Bind Gradio to all interfaces so -p port mapping works (app.py
# defaults to 127.0.0.1 outside the container).
ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 7860

CMD ["python", "app.py"]
