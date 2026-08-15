#!/usr/bin/env bash
set -uo pipefail

CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
FAILURES=0
WARNINGS=0

pass() { printf '[PASS] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
warn() { printf '[WARN] %s\n' "$1"; WARNINGS=$((WARNINGS + 1)); }

if grep -qi microsoft /proc/version 2>/dev/null && [ "${WSL_INTEROP:-}" != "" ]; then
  pass "当前环境为 WSL2"
else
  fail "未检测到 WSL2；请从 Ubuntu-24.04 WSL2 运行 make doctor"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE="$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>&1 | head -1)"
  if [ -n "$GPU_LINE" ]; then pass "WSL NVIDIA 映射可用：$GPU_LINE"; else fail "nvidia-smi 未返回 GPU"; fi
else
  fail "WSL 中找不到 nvidia-smi；更新 Windows NVIDIA 驱动，不要安装 Linux 显示驱动"
fi

if ! command -v docker >/dev/null 2>&1; then
  fail "找不到 docker CLI；在 Docker Desktop > Settings > Resources > WSL Integration 启用当前发行版"
elif ! docker info >/dev/null 2>&1; then
  fail "Docker daemon 不可用；启动 Docker Desktop 并启用 WSL Integration"
else
  pass "Docker daemon 可用"
  if docker image inspect "$CUDA_IMAGE" >/dev/null 2>&1 || docker pull "$CUDA_IMAGE"; then
    DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$CUDA_IMAGE" 2>/dev/null || true)"
    pass "CUDA 镜像可用：${DIGEST:-$CUDA_IMAGE（本地镜像暂无 RepoDigest）}"
    CONTAINER_NVCC="$(docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --security-opt seccomp=builtin "$CUDA_IMAGE" nvcc --version 2>&1 | tail -1)"
    if [ -n "$CONTAINER_NVCC" ]; then
      pass "固定容器 NVCC：$CONTAINER_NVCC"
    else
      fail "固定 CUDA 容器内找不到 NVCC"
    fi
    if GPU_INFO="$(docker run --rm --gpus device=0 --network none --read-only --cap-drop ALL --security-opt no-new-privileges --security-opt seccomp=builtin "$CUDA_IMAGE" nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>&1)"; then
      pass "CUDA 容器识别 GPU：$(printf '%s' "$GPU_INFO" | head -1)"
    else
      fail "CUDA 容器无法访问 GPU：$GPU_INFO"
    fi
  else
    fail "无法拉取固定 CUDA 镜像 $CUDA_IMAGE"
  fi
fi

if command -v nvcc >/dev/null 2>&1; then
  NVCC_LINE="$(nvcc --version | tail -1)"
  pass "WSL NVCC 可用：$NVCC_LINE"
else
  warn "WSL 主机没有 nvcc（平台正常运行只要求固定 CUDA 容器内有 nvcc）"
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  cat >"$TMP_DIR/doctor.cu" <<'CUDA'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void ping(int* value) { *value = 42; }
int main() {
  int *device = nullptr, host = 0, runtime_version = 0, driver_api_version = 0;
  if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) return 1;
  if (cudaDriverGetVersion(&driver_api_version) != cudaSuccess) return 1;
  if (cudaMalloc(&device, sizeof(int)) != cudaSuccess) return 2;
  ping<<<1, 1>>>(device);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  if (cudaMemcpy(&host, device, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
  cudaFree(device);
  std::printf("value=%d runtime=%d driver_api=%d\\n", host, runtime_version, driver_api_version);
  return host == 42 ? 0 : 5;
}
CUDA
  chmod 0777 "$TMP_DIR"
  if docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --security-opt seccomp=builtin --user 65534:65534 --tmpfs /tmp:rw,nosuid,nodev,size=64m --mount "type=bind,src=$TMP_DIR,dst=/work" "$CUDA_IMAGE" nvcc -arch=native /work/doctor.cu -o /work/doctor >/tmp/myleetgpu-doctor-compile.log 2>&1 \
    && RESULT="$(docker run --rm --gpus device=0 --network none --read-only --cap-drop ALL --security-opt no-new-privileges --security-opt seccomp=builtin --user 65534:65534 --tmpfs /tmp:rw,nosuid,nodev,size=64m --mount "type=bind,src=$TMP_DIR,dst=/work,readonly" "$CUDA_IMAGE" /work/doctor 2>&1)" \
    && [[ "$RESULT" == value=42\ runtime=* ]]; then
    pass "最小 CUDA 程序在真实 GPU 上编译并运行：$RESULT"
  else
    fail "最小 CUDA 程序失败：$(tail -8 /tmp/myleetgpu-doctor-compile.log 2>/dev/null || true) ${RESULT:-}"
  fi
fi

printf '\nDoctor summary: %d failure(s), %d warning(s)\n' "$FAILURES" "$WARNINGS"
[ "$FAILURES" -eq 0 ]
