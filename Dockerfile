# llama-swap — amd64 / CUDA-only
#
# Bundles llama.cpp, WhisperLive (faster-whisper), and Parakeet (ONNX ASR),
# all orchestrated by llama-swap.
#
# Build:
#   docker buildx build -t llama-swap .
#
# Override CUDA architectures for your GPU (default covers most Turing/Ampere/Ada):
#   docker buildx build --build-arg CMAKE_CUDA_ARCHITECTURES="89" -t llama-swap .

# ── Version pins ───────────────────────────────────────────────────────────────
# Renovate manages these via customManagers in renovate.json.
# The COPY --from=ghcr.io/astral-sh/uv:* lines below are tracked by
# Renovate's built-in dockerfile manager.

ARG LLAMA_VERSION=b8996
ARG LS_VERSION=v212
ARG WHISPERLIVE_VERSION=v0.8.0
ARG CMAKE_CUDA_ARCHITECTURES="75;86;89;90;100"

# ── Builder base ───────────────────────────────────────────────────────────────

FROM nvidia/cuda:13.2.1-devel-ubuntu24.04 AS builder-base

ENV DEBIAN_FRONTEND=noninteractive
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=2G
ENV PATH="/usr/lib/ccache:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential cmake git libssl-dev curl ca-certificates ccache \
  && rm -rf /var/lib/apt/lists/*

# ── Build llama.cpp ────────────────────────────────────────────────────────────

FROM builder-base AS llama-build

ARG LLAMA_VERSION
ARG CMAKE_CUDA_ARCHITECTURES

RUN mkdir -p /src/llama.cpp \
  && cd /src/llama.cpp \
  && git init \
  && git remote add origin https://github.com/ggml-org/llama.cpp.git \
  && git fetch --depth=1 origin "${LLAMA_VERSION}" \
  && git checkout FETCH_HEAD

RUN --mount=type=cache,id=ccache,target=/ccache \
  --mount=type=cache,id=llama-build,target=/src/llama.cpp/build \
  rm -rf /src/llama.cpp/build/CMakeCache.txt /src/llama.cpp/build/CMakeFiles \
  && cmake -S /src/llama.cpp -B /src/llama.cpp/build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
  -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -lcuda" \
  && cmake --build /src/llama.cpp/build --config Release -j"$(nproc)" \
  --target llama-server llama-cli \
  && mkdir -p /install/bin \
  && cp /src/llama.cpp/build/bin/llama-server /install/bin/ \
  && cp /src/llama.cpp/build/bin/llama-cli /install/bin/

# ── Download llama-swap ────────────────────────────────────────────────────────

FROM ubuntu:24.04 AS llama-swap-download

ARG LS_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
  curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN LS_NUM="${LS_VERSION#v}" \
  && mkdir -p /install/bin \
  && curl -fsSL \
  "https://github.com/mostlygeek/llama-swap/releases/download/${LS_VERSION}/llama-swap_${LS_NUM}_linux_amd64.tar.gz" \
  -o /tmp/llama-swap.tar.gz \
  && tar -xzf /tmp/llama-swap.tar.gz -C /install/bin \
  && chmod +x /install/bin/llama-swap \
  && echo "${LS_VERSION}" > /install/llama-swap-version

# ── Build WhisperLive venv ─────────────────────────────────────────────────────

FROM ubuntu:24.04 AS whisperlive-build

ARG WHISPERLIVE_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=3.12
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN apt-get update && apt-get install -y --no-install-recommends \
  git ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv

RUN uv python install 3.12

RUN git clone --depth 1 --branch "${WHISPERLIVE_VERSION}" \
  https://github.com/collabora/WhisperLive /opt/WhisperLive

RUN --mount=type=cache,id=whisperlive-uv,target=/root/.cache/uv \
  uv venv --python 3.12 /opt/WhisperLive/.venv \
  && uv pip install --python /opt/WhisperLive/.venv \
  -r /opt/WhisperLive/requirements/server.txt \
  && uv pip install --python /opt/WhisperLive/.venv \
  --no-deps /opt/WhisperLive

# ── Runtime ────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04 AS runtime

ARG LLAMA_VERSION
ARG LS_VERSION
ARG WHISPERLIVE_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PATH="/usr/local/bin:${PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
  libgomp1 ffmpeg curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# CUDA stub drivers for container compatibility
COPY --from=builder-base /usr/local/cuda/lib64/stubs/libcuda.so \
  /usr/local/cuda/lib64/stubs/libcuda.so
COPY --from=builder-base /usr/local/cuda/lib64/stubs/libcuda.so \
  /usr/local/cuda/lib64/stubs/libcuda.so.1

# llama.cpp binaries
COPY --from=llama-build /install/bin/llama-server /usr/local/bin/
COPY --from=llama-build /install/bin/llama-cli /usr/local/bin/

# llama-swap binary
COPY --from=llama-swap-download /install/bin/llama-swap /usr/local/bin/
COPY --from=llama-swap-download /install/llama-swap-version /tmp/

# uv-managed Python interpreters (venv symlinks resolve against these)
COPY --from=whisperlive-build /opt/uv-python /opt/uv-python

# WhisperLive (source tree includes .venv)
COPY --from=whisperlive-build /opt/WhisperLive /opt/WhisperLive

RUN ldconfig \
  && mkdir -p /etc/llama-swap/config /models

COPY config.example.yaml /etc/llama-swap/config/config.yaml

RUN echo "llama.cpp: ${LLAMA_VERSION}" > /versions.txt \
  && echo "llama-swap: $(cat /tmp/llama-swap-version)" >> /versions.txt \
  && echo "whisperlive: ${WHISPERLIVE_VERSION}" >> /versions.txt \
  && echo "build_timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /versions.txt

WORKDIR /models

ENTRYPOINT ["llama-swap"]
CMD ["-config", "/etc/llama-swap/config/config.yaml", "-listen", "0.0.0.0:8080"]
