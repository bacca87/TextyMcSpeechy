# TextyMcSpeechy training image aligned to OHF-Voice/piper1-gpl (piper.train, Lightning 2.x)
# Tested stack (smoke test, 2026-08): torch 2.13+cu130, lightning 2.6.5, cython 3.x,
# espeakbridge built from source (bundled espeak-ng), monotonic_align built.
FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04 AS base
ARG USERNAME=nonrootuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Set working directory
WORKDIR /app

# System dependencies + piper1-gpl pinned to the exact commit used for the smoke test
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    git espeak-ng tmux ffmpeg inotify-tools \
    build-essential cmake ninja-build && \
    git init piper && cd piper && git fetch --depth 1 https://github.com/OHF-Voice/piper1-gpl.git ffb62233b04dd0b04f005cd7c1f879eb9b0889fd && git checkout FETCH_HEAD && \
    rm -rf /var/lib/apt/lists/*

# Virtual environment + training stack (torch first, then the rest of [train] extras)
WORKDIR /app/piper
RUN python3.12 -m venv .venv && \
    /app/piper/.venv/bin/pip install --upgrade pip wheel setuptools && \
    /app/piper/.venv/bin/pip install torch==2.13.0 && \
    /app/piper/.venv/bin/pip install -e ".[train]" onnxscript torchaudio scikit-build && \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ /app/piper/.venv/bin/python setup.py build_ext --inplace && \
    bash /app/piper/build_monotonic_align.sh

# create non root user (ubuntu24.04 base already ships uid/gid 1000: reuse them)
WORKDIR /
RUN if ! getent group $USER_GID >/dev/null; then groupadd --gid $USER_GID $USERNAME; fi && \
    if ! getent passwd $USER_UID >/dev/null; then \
        useradd -m -s /bin/bash -u $USER_UID -g $USER_GID $USERNAME; \
    else \
        usermod -l $USERNAME -d /home/$USERNAME -m "$(getent passwd $USER_UID | cut -d: -f1)"; \
    fi && \
    chown -R $USER_UID:$USER_GID /home/$USERNAME

# Environment
ENV PATH="/app/piper/.venv/bin:$PATH"
ENV CUDA_VISIBLE_DEVICES=0

# Mount volume
VOLUME ["/app/tts_dojo"]

# Default command
CMD ["/bin/bash"]
