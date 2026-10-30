# syntax=docker/dockerfile:1
# Hypnos with CUDA engines. Build: docker build -t hypnos .   (published as ghcr.io/advancedlogic/hypnos)
#
#   docker run --gpus all -p 8080:8080 -v ./models:/opt/hypnos/models ghcr.io/advancedlogic/hypnos
#
# Models are mounted at /opt/hypnos/models (download them with the image's fetch script, see the
# README). The config is /opt/hypnos/hypnos.toml: mount your own over it, or point HYPNOS_CONFIG
# at another mounted file. HYPNOS_API_KEYS sets the API keys.

ARG CUDA_VERSION=12.8.1
# 24.04 or newer: ort's prebuilt static ONNX Runtime needs glibc 2.38+ and GCC 14's libstdc++.
ARG UBUNTU=ubuntu24.04

# --- playground --------------------------------------------------------------------------------
FROM node:22-bookworm-slim AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- engines + server --------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS build
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
# CMAKE_CUDA_ARCHITECTURES is left to ggml's portable default (as in the release archives).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cd backend && cargo build --release -p hypnos-server ${JOBS:+-j "$JOBS"}
COPY --from=frontend /src/frontend/dist frontend/dist
RUN ./backend/scripts/package-release.sh linux-x64-cuda --version "$VERSION" --out /dist \
    && mv "/dist/hypnos-$VERSION-linux-x64-cuda" /dist/hypnos

# --- runtime -----------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
LABEL org.opencontainers.image.title="Hypnos" \
      org.opencontainers.image.description="One OpenAI-compatible endpoint for every modality on one GPU" \
      org.opencontainers.image.source="https://github.com/advancedlogic/hypnos" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /dist/hypnos /opt/hypnos
WORKDIR /opt/hypnos
# Listen on every interface inside the container; publish the port with -p. Without API keys the
# server logs a warning, since anything that can reach the published port can use it.
RUN sed -i 's/^host = "127.0.0.1"/host = "0.0.0.0"/' hypnos.toml && grep -q '^host = "0.0.0.0"' hypnos.toml
VOLUME ["/opt/hypnos/models"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m CMD curl -fsS http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["/opt/hypnos/hypnos"]
