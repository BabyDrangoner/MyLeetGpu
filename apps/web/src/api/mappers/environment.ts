import type { EnvironmentSnapshot, KernelLanguage } from '../../domain/types'
import { asDate, asKernelLanguage, asRecord, asString } from './common'

export function normalizeEnvironment(value: unknown, fallbackLanguage: KernelLanguage = 'cuda_cpp'): EnvironmentSnapshot {
  const raw = asRecord(value)
  const telemetry = asRecord(raw.telemetry)
  const toolchain = asRecord(raw.toolchain)
  const unavailable = Array.isArray(raw.unavailable_metrics)
    ? raw.unavailable_metrics.map((item) => asString(item))
    : Object.entries(telemetry).filter(([, item]) => item === null || item === undefined || item === '').map(([key]) => key)
  const healthy = typeof raw.healthy === 'boolean' ? raw.healthy : raw.status === 'healthy'
  return {
    ...raw,
    healthy,
    execution_target: raw.execution_target === 'cpu' ? 'cpu' : raw.execution_target === 'colab' ? 'colab' : raw.execution_target === 'local' ? 'local' : undefined,
    backend: asKernelLanguage(raw.backend, fallbackLanguage),
    status: asString(raw.status, healthy ? 'healthy' : 'unhealthy'),
    gpu_name: asString(raw.gpu_name ?? raw.gpu) || undefined,
    cpu_name: asString(raw.cpu_name ?? toolchain.cpu_name) || undefined,
    platform: asString(raw.platform ?? toolchain.platform) || undefined,
    architecture: asString(raw.architecture ?? toolchain.architecture) || undefined,
    compiler_version: asString(raw.compiler_version ?? toolchain.compiler_version) || undefined,
    compute_capability: asString(raw.compute_capability) || undefined,
    driver_version: asString(raw.driver_version) || undefined,
    cuda_runtime_version: asString(raw.cuda_runtime_version ?? raw.cuda_version) || undefined,
    nvcc_version: asString(raw.nvcc_version) || undefined,
    python_version: asString(raw.python_version ?? toolchain.python_version) || undefined,
    torch_version: asString(raw.torch_version ?? toolchain.torch_version) || undefined,
    torch_cuda_version: asString(raw.torch_cuda_version ?? toolchain.torch_cuda_version) || undefined,
    triton_version: asString(raw.triton_version ?? toolchain.triton_version) || undefined,
    container_image: asString(raw.container_image ?? raw.cuda_image) || undefined,
    container_digest: asString(raw.container_digest ?? raw.image_digest) || undefined,
    fingerprint: asString(raw.fingerprint) || undefined,
    checked_at: asDate(raw.checked_at ?? raw.observed_at ?? raw.created_at) || undefined,
    unavailable_metrics: unavailable,
    message: asString(raw.message ?? raw.error) || undefined,
    telemetry: telemetry as Record<string, string | null>,
    toolchain: toolchain as Record<string, string | null>,
  }
}
