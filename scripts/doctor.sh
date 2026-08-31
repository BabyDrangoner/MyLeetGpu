#!/usr/bin/env bash
set -uo pipefail

CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
TRITON_IMAGE="${TRITON_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014}"
FAILURES=0
WARNINGS=0
CUDA_ARCH=""
TRITON_READY=0
TORCH_READY=0

pass() { printf '[PASS] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
warn() { printf '[WARN] %s\n' "$1"; WARNINGS=$((WARNINGS + 1)); }

if grep -qi microsoft /proc/version 2>/dev/null && [ "${WSL_INTEROP:-}" != "" ]; then
  pass "当前环境为 WSL2"
else
  fail "未检测到 WSL2；请从 Ubuntu-24.04 WSL2 运行 make doctor"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if GPU_OUTPUT="$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>&1)"; then
    GPU_LINE="$(printf '%s' "$GPU_OUTPUT" | head -1)"
    COMPUTE_CAPABILITY="$(printf '%s' "$GPU_LINE" | awk -F, '{value=$NF; gsub(/[[:space:].]/, "", value); print value}')"
    if [ -n "$GPU_LINE" ] && [[ "$COMPUTE_CAPABILITY" =~ ^[0-9]+$ ]]; then
      CUDA_ARCH="sm_${COMPUTE_CAPABILITY}"
      pass "WSL NVIDIA 映射可用：$GPU_LINE（编译架构 $CUDA_ARCH）"
    else
      fail "nvidia-smi 未返回可解析的 GPU Compute Capability：$GPU_LINE"
    fi
  else
    fail "nvidia-smi 查询失败：$GPU_OUTPUT"
  fi
else
  fail "WSL 中找不到 nvidia-smi；更新 Windows NVIDIA 驱动，不要安装 Linux 显示驱动"
fi

if ! command -v docker >/dev/null 2>&1; then
  fail "找不到 docker CLI；在 Docker Desktop > Settings > Resources > WSL Integration 启用当前发行版"
elif ! docker info >/dev/null 2>&1; then
  fail "Docker daemon 不可用；启动 Docker Desktop 并启用 WSL Integration"
else
  pass "Docker daemon 可用"
  if COMPOSE_VERSION="$(docker compose version --short 2>/dev/null)"; then
    pass "Docker Compose v2 可用：$COMPOSE_VERSION"
  else
    fail "Docker Compose v2 不可用；请更新 Docker Desktop"
  fi
  if docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
    pass "NVIDIA Container Runtime 已注册"
  else
    fail "Docker 未注册 NVIDIA runtime；检查 Docker Desktop GPU/WSL 集成"
  fi
  if docker image inspect "$CUDA_IMAGE" >/dev/null 2>&1 || docker pull "$CUDA_IMAGE"; then
    DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$CUDA_IMAGE" 2>/dev/null || true)"
    pass "CUDA 镜像可用：${DIGEST:-$CUDA_IMAGE（本地镜像暂无 RepoDigest）}"
    if CONTAINER_NVCC_OUTPUT="$(docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges "$CUDA_IMAGE" nvcc --version 2>&1)"; then
      CONTAINER_NVCC="$(printf '%s' "$CONTAINER_NVCC_OUTPUT" | tail -1)"
      pass "固定容器 NVCC：$CONTAINER_NVCC"
    else
      fail "固定 CUDA 容器内 NVCC 检查失败：$CONTAINER_NVCC_OUTPUT"
    fi
    if GPU_INFO="$(docker run --rm --gpus device=0 --network none --read-only --cap-drop ALL --security-opt no-new-privileges "$CUDA_IMAGE" nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>&1)"; then
      CONTAINER_GPU_LINE="$(printf '%s' "$GPU_INFO" | tail -1)"
      if [ -n "$CONTAINER_GPU_LINE" ]; then
        pass "CUDA 容器识别 GPU：$CONTAINER_GPU_LINE"
      else
        fail "CUDA 容器 GPU 查询没有返回设备信息"
      fi
    else
      fail "CUDA 容器无法访问 GPU：$GPU_INFO"
    fi
  else
    fail "无法拉取固定 CUDA 镜像 $CUDA_IMAGE"
  fi
  if docker image inspect "$TRITON_IMAGE" >/dev/null 2>&1 || docker pull "$TRITON_IMAGE"; then
    TRITON_DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$TRITON_IMAGE" 2>/dev/null || true)"
    if TORCH_INFO="$(docker run --rm --gpus device=0 --network none --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,nosuid,nodev,noexec,size=512m --env HOME=/tmp --entrypoint python "$TRITON_IMAGE" -I -B -c 'import platform, torch; print(f"Python {platform.python_version()} / Torch {torch.__version__} / CUDA {torch.version.cuda} / GPU {torch.cuda.get_device_name(0)}")' 2>&1)"; then
      pass "PyTorch 镜像可用：${TRITON_DIGEST:-$TRITON_IMAGE}"
      pass "PyTorch 工具链：$TORCH_INFO"
      TORCH_READY=1
    else
      fail "固定 PyTorch 容器无法加载 GPU 工具链：$TORCH_INFO"
    fi
    if TRITON_INFO="$(docker run --rm --gpus device=0 --network none --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,nosuid,nodev,exec,size=512m --env HOME=/tmp --entrypoint python "$TRITON_IMAGE" -I -B -c 'import platform, torch, triton; print(f"Python {platform.python_version()} / Torch {torch.__version__} / Triton {triton.__version__} / CUDA {torch.version.cuda} / GPU {torch.cuda.get_device_name(0)}")' 2>&1)"; then
      pass "Triton 镜像可用：${TRITON_DIGEST:-$TRITON_IMAGE}"
      pass "Triton 工具链：$TRITON_INFO"
      TRITON_READY=1
    else
      fail "固定 Triton 容器无法加载 GPU 工具链：$TRITON_INFO"
    fi
  else
    fail "无法拉取固定 Triton 镜像 $TRITON_IMAGE"
  fi
fi

if command -v nvcc >/dev/null 2>&1; then
  NVCC_LINE="$(nvcc --version | tail -1)"
  pass "WSL NVCC 可用：$NVCC_LINE"
else
  warn "WSL 主机没有 nvcc（平台正常运行只要求固定 CUDA 容器内有 nvcc）"
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && [ -n "$CUDA_ARCH" ]; then
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
  std::printf("value=%d runtime=%d driver_api=%d\n", host, runtime_version, driver_api_version);
  return host == 42 ? 0 : 5;
}
CUDA
  cat >"$TMP_DIR/triton_doctor.py" <<'PYTHON'
import torch
import triton
import triton.language as tl


@triton.jit
def add_one(input_ptr, output_ptr, n: tl.constexpr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, values + 1.0, mask=mask)


n = 257
input_tensor = torch.arange(n, device="cuda", dtype=torch.float32)
output_tensor = torch.empty_like(input_tensor)
add_one[(1,)](input_tensor, output_tensor, n, block=512)
torch.cuda.synchronize()
if not torch.equal(output_tensor, input_tensor + 1.0):
    raise SystemExit("wrong Triton result")
print(f"value={output_tensor[-1].item():.0f} triton={triton.__version__} torch={torch.__version__}")
PYTHON
  cat >"$TMP_DIR/torch_doctor.py" <<'PYTHON'
import torch


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True)
query = torch.arange(64, device="cuda", dtype=torch.float32).reshape(1, 2, 4, 8) / 64
key = torch.flip(query, dims=(-2,))
value = query + 0.25
scores = torch.matmul(query, key.transpose(-2, -1)) * (query.shape[-1] ** -0.5)
output = torch.matmul(torch.softmax(scores, dim=-1), value)
torch.cuda.synchronize()
if output.shape != query.shape or not bool(torch.all(torch.isfinite(output))):
    raise SystemExit("wrong PyTorch attention result")
print(f"shape={tuple(output.shape)} torch={torch.__version__} cuda={torch.version.cuda}")
PYTHON
  chmod 0777 "$TMP_DIR"
  if docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 --tmpfs /tmp:rw,nosuid,nodev,size=64m --mount "type=bind,src=$TMP_DIR,dst=/work" "$CUDA_IMAGE" nvcc -arch="$CUDA_ARCH" /work/doctor.cu -o /work/doctor >"$TMP_DIR/compile.log" 2>&1 \
    && RESULT="$(docker run --rm --gpus device=0 --network none --read-only --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 --tmpfs /tmp:rw,nosuid,nodev,size=64m --mount "type=bind,src=$TMP_DIR,dst=/work,readonly" "$CUDA_IMAGE" /work/doctor 2>&1)" \
    && RESULT_LINE="$(printf '%s' "$RESULT" | grep -m1 '^value=42 runtime=')" \
    && [ -n "$RESULT_LINE" ]; then
    pass "最小 CUDA 程序在真实 GPU 上编译并运行：$RESULT_LINE"
  else
    fail "最小 CUDA 程序失败：$(tail -8 "$TMP_DIR/compile.log" 2>/dev/null || true) ${RESULT:-}"
  fi
  if [ "$TORCH_READY" -eq 1 ]; then
    if TORCH_RESULT="$(docker run --rm --gpus device=0 --network none --read-only --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=512m --env HOME=/tmp --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env NVIDIA_TF32_OVERRIDE=0 --mount "type=bind,src=$TMP_DIR,dst=/work,readonly" --entrypoint python "$TRITON_IMAGE" -I -B /work/torch_doctor.py 2>&1)"; then
      pass "最小 PyTorch attention 在真实 GPU 上运行：$TORCH_RESULT"
    else
      fail "最小 PyTorch attention 失败：$TORCH_RESULT"
    fi
  fi
  if [ "$TRITON_READY" -eq 1 ]; then
    if TRITON_RESULT="$(docker run --rm --gpus device=0 --network none --read-only --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 --tmpfs /tmp:rw,nosuid,nodev,exec,size=512m --env HOME=/tmp --env TRITON_CACHE_DIR=/tmp/triton-cache --mount "type=bind,src=$TMP_DIR,dst=/work,readonly" --entrypoint python "$TRITON_IMAGE" -I -B /work/triton_doctor.py 2>&1)"; then
      pass "最小 Triton kernel 在真实 GPU 上 JIT 并运行：$TRITON_RESULT"
    else
      fail "最小 Triton kernel 失败：$TRITON_RESULT"
    fi
  fi
fi

printf '\nDoctor summary: %d failure(s), %d warning(s)\n' "$FAILURES" "$WARNINGS"
[ "$FAILURES" -eq 0 ]
