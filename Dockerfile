# syntax=docker/dockerfile:1
# Hypnos with CUDA engines. Build: docker build -t hypnos .   (published as ghcr.io/advancedlogic/hypnos)
# From a release archive instead: put it in prebuilt/ and add --build-arg SERVER=archive.
#
#   docker run --gpus all -p 9999:9999 -v ./models:/opt/hypnos/models ghcr.io/advancedlogic/hypnos
#
# Models are mounted at /opt/hypnos/models (download them with the image's fetch script, see the
# README). The config is /opt/hypnos/hypnos.toml: mount your own over it, or point HYPNOS_CONFIG
# at another mounted file. HYPNOS_API_KEYS sets the API keys.

ARG CUDA_VERSION=12.8.1
# 24.04 or newer: ort's prebuilt static ONNX Runtime needs glibc 2.38+ and GCC 14's libstdc++.
ARG UBUNTU=ubuntu24.04
# DOCLING_CUDA=1 adds the CUDA 13 runtime and cuDNN 9 to the image (about 1.5 GB) so a docling
# model with device = "gpu" can run; the provider library itself is always in the image. The
# default image converts documents on the CPU, which needs none of this.
ARG DOCLING_CUDA=0
# Where the server comes from: "source" compiles the engines in the image (an hour or more on a
# CUDA build); "archive" packs a linux-x64-cuda release archive placed in prebuilt/ (what
# release.yml does with the archive its build job just made, built with the same CUDA and Ubuntu).
ARG SERVER=source

# --- playground --------------------------------------------------------------------------------
FROM node:22-bookworm-slim AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- engines + server --------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS server-source
ARG VERSION=docker
# cargo -j: also the --parallel of each engine's CMake build, so up to JOBS^2 nvcc processes run at
# once. Some CUDA kernels need several GB each; 2 fits a 16 GB machine, unset uses every core.
ARG JOBS=2
ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH \
    CMAKE_GENERATOR=Ninja \
    LLAMA_LIB_PROFILE=Release
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev libclang-dev clang git curl ca-certificates ninja-build python3 \
    && rm -rf /var/lib/apt/lists/*
# Ubuntu 24.04 ships CMake 3.28; the engines are tested with a current one.
RUN curl -fsSL "https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-$(uname -m).tar.gz" \
        | tar -xz --strip-components=1 -C /usr/local
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable --component rustfmt
WORKDIR /src
COPY . .
RUN ./backend/scripts/vendor.sh
# The GPUs the release archives are built for (see build-server.yml); override with --build-arg.
ARG CUDA_ARCHITECTURES="75-virtual;80-virtual;86-real;89-real;120a-real"
ENV CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cd backend && cargo build --release -p hypnos-server --features docling-cuda ${JOBS:+-j "$JOBS"}
COPY --from=frontend /src/frontend/dist frontend/dist
RUN ./backend/scripts/package-release.sh linux-x64-cuda --version "$VERSION" --out /dist \
    && mv "/dist/hypnos-$VERSION-linux-x64-cuda" /dist/hypnos

# --- or a prebuilt archive -----------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU} AS server-archive
COPY prebuilt/hypnos-*-linux-x64-cuda.tar.gz /tmp/
RUN set -e; n=$(ls /tmp/hypnos-*-linux-x64-cuda.tar.gz | wc -l); \
    [ "$n" = 1 ] || { echo "prebuilt/ must hold exactly one linux-x64-cuda archive, found $n" >&2; exit 1; }; \
    mkdir -p /dist/hypnos && tar -xzf /tmp/hypnos-*-linux-x64-cuda.tar.gz -C /dist/hypnos --strip-components=1 \
    && rm /tmp/hypnos-*.tar.gz && test -x /dist/hypnos/hypnos

FROM server-${SERVER} AS server

# --- runtime -----------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
LABEL org.opencontainers.image.title="Hypnos" \
      org.opencontainers.image.description="One OpenAI-compatible endpoint for every modality on one GPU" \
      org.opencontainers.image.source="https://github.com/advancedlogic/hypnos" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"
# Fonts: docling's page renderer substitutes them for the standard PDF fonts most documents
# reference without embedding; the CUDA runtime image ships none.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
        fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
# The CUDA images carry NVIDIA's apt repository, which also serves the CUDA 13 packages.
ARG DOCLING_CUDA
RUN if [ "$DOCLING_CUDA" = 1 ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            cuda-cudart-13-0 libcublas-13-0 libcudnn9-cuda-13 \
        && rm -rf /var/lib/apt/lists/*; \
    fi
COPY --from=server /dist/hypnos /opt/hypnos
WORKDIR /opt/hypnos
# Listen on every interface inside the container; publish the port with -p. Without API keys the
# server logs a warning, since anything that can reach the published port can use it.
RUN sed -i 's/^host = "127.0.0.1"/host = "0.0.0.0"/' hypnos.toml && grep -q '^host = "0.0.0.0"' hypnos.toml
VOLUME ["/opt/hypnos/models"]
EXPOSE 9999
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m CMD curl -fsS http://127.0.0.1:9999/health || exit 1
ENTRYPOINT ["/opt/hypnos/hypnos"]
