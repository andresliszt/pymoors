# syntax=docker/dockerfile:1

FROM mcr.microsoft.com/devcontainers/base:ubuntu

ARG DEBIAN_FRONTEND=noninteractive


# RUN <<EOF
#   apt-get update
#   apt-get install -y --no-install-recommends \
    
#   rm -rf /var/lib/apt/lists/*
# EOF

RUN <<EOF
 curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y
 curl -LsSf https://astral.sh/uv/install.sh | sh
EOF
