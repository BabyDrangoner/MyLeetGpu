import type { Page, Route } from '@playwright/test'

export const starterSource = `#include <cuda_runtime.h>

__global__ void solve_kernel(const float* a, const float* b, float* output, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) output[index] = a[index] + b[index];
}

void solve(const float* a, const float* b, float* output, int n, cudaStream_t stream) {
  solve_kernel<<<(n + 255) / 256, 256, 0, stream>>>(a, b, output, n);
}
`

export const tritonStarterSource = `import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(a, b, output, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    tl.store(output + offsets, tl.load(a + offsets, mask=mask) + tl.load(b + offsets, mask=mask), mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, output: torch.Tensor, n: int) -> None:
    add_kernel[(triton.cdiv(n, 256),)](a, b, output, n, BLOCK_SIZE=256)
`

const environment = {
  id: 'env-1',
  healthy: true,
  status: 'healthy',
  fingerprint: 'rtx4060-sm89-cuda126-test',
  gpu_name: 'NVIDIA GeForce RTX 4060',
  compute_capability: '8.9',
  driver_version: '591.74',
  cuda_runtime_version: '12.6',
  nvcc_version: '12.6.85',
  python_version: '3.11.10',
  torch_version: '2.5.1+cu124',
  triton_version: '3.2.0',
  cuda_image: 'nvidia/cuda:12.6.3-devel-ubuntu22.04',
  image_digest: 'sha256:e2e',
  observed_at: '2026-08-16T01:00:00Z',
  telemetry: { temperature_c: null, power_w: null },
}

const problems = [
  { slug: 'vector-addition', title: '向量逐元素相加', difficulty: 'easy', revision: '1', summary: '实现高吞吐的逐元素相加。', languages: ['cuda_cpp', 'triton_python'] },
  { slug: 'matrix-transpose', title: '行主序矩阵转置', difficulty: 'medium', revision: '1', summary: '使用共享内存完成矩阵转置。', languages: ['cuda_cpp', 'triton_python'] },
  { slug: 'reduction', title: '单精度向量求和归约', difficulty: 'hard', revision: '1', summary: '实现稳定的并行归约。', languages: ['cuda_cpp', 'triton_python'] },
]

function problemDetail(slug: string) {
  const summary = problems.find((item) => item.slug === slug) ?? problems[0]
  return {
    ...summary,
    default_language: 'cuda_cpp',
    implementations: {
      cuda_cpp: { language: 'cuda_cpp', display_name: 'CUDA C++', source_suffix: '.cu', editor_language: 'cpp', starter_code: starterSource, signature: { symbol: 'solve', declaration: 'void solve(const float* a, const float* b, float* output, int n, cudaStream_t stream)' } },
      triton_python: { language: 'triton_python', display_name: 'Triton (Python)', source_suffix: '.py', editor_language: 'python', starter_code: tritonStarterSource, signature: { symbol: 'solve', declaration: 'def solve(a: torch.Tensor, b: torch.Tensor, output: torch.Tensor, n: int) -> None' } },
    },
    statement_markdown: '## 任务\n\n实现平台声明的 `solve` 接口。\n\n- 使用传入 stream\n- 不得越界访问',
    starter_code: starterSource,
    signature: { symbol: 'solve', declaration: 'void solve(const float* a, const float* b, float* output, int n, cudaStream_t stream)' },
    constraints: { n: { min: 1, max: 16_777_216 }, aliasing: 'none' },
    benchmark: { protocol_version: '1', sizes: [{ label: '64K' }, { label: '1M' }], warmup: 8, iterations: 20 },
  }
}

function savedVersion(id: string, name: string, medianScale: number, fingerprint = environment.fingerprint, language: 'cuda_cpp' | 'triton_python' = 'cuda_cpp') {
  return {
    id,
    problem_id: 'vector-addition',
    problem_revision: '1',
    language,
    name,
    notes: id === 'v1' ? '直接加载实现' : '使用向量化访存',
    source_hash: id.repeat(64).slice(0, 64),
    source_code: language === 'triton_python' ? tritonStarterSource : id === 'v1' ? starterSource : starterSource.replace('const int index', '// float4 path\n  const int index'),
    created_at: '2026-08-16T01:00:00Z',
    correctness_status: 'passed',
    compile_flags: language === 'triton_python' ? ['triton-autotune=off'] : ['--std=c++17', '-O3'],
    benchmark_runs: [{
      id: `run-${id}`,
      suite_hash: 'suite-e2e',
      protocol_version: '1',
      compile_flags: language === 'triton_python' ? ['triton-autotune=off'] : ['--std=c++17', '-O3'],
      input_sizes: ['64K', '1M'],
      seed: 424242,
      warmup: 8,
      iterations: 20,
      environment: { ...environment, fingerprint },
      measurements: [
        { size: '64K', median_ms: 0.08 * medianScale, p95_ms: 0.09 * medianScale, min_ms: 0.075 * medianScale, cv: 0.03, samples_ms: Array(20).fill(0.08 * medianScale) },
        { size: '1M', median_ms: 0.52 * medianScale, p95_ms: 0.56 * medianScale, min_ms: 0.5 * medianScale, cv: 0.02, samples_ms: Array(20).fill(0.52 * medianScale) },
      ],
      created_at: '2026-08-16T01:00:00Z',
    }],
  }
}

export interface MockApiState {
  versions: ReturnType<typeof savedVersion>[]
  submittedJobs: Array<Record<string, unknown>>
  deleteConfirmed: boolean
}

export async function installMockApi(page: Page, options: { versions?: boolean; comparable?: boolean; tritonVersion?: boolean } = {}) {
  const initialVersions = options.versions ? [savedVersion('v1', '直接加载', 1), savedVersion('v2', '向量化 float4', 0.72, options.comparable === false ? 'different-env' : environment.fingerprint)] : []
  if (options.tritonVersion) initialVersions.push(savedVersion('t1', 'Triton baseline', 0.9, environment.fingerprint, 'triton_python'))
  const state: MockApiState = {
    versions: initialVersions,
    submittedJobs: [],
    deleteConfirmed: false,
  }
  const jobs = new Map<string, Record<string, unknown>>()
  const drafts = new Map<string, string>()

  await page.route(/^https?:\/\/[^/]+\/api\/.*/, async (route: Route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const method = request.method()
    const fulfill = (payload: unknown, status = 200) => route.fulfill({ status, contentType: 'application/json', body: payload === undefined ? '' : JSON.stringify(payload) })

    if (path === '/api/environment' && method === 'GET') return fulfill({ ...environment, backend: url.searchParams.get('language') ?? 'cuda_cpp' })
    if (path === '/api/problems' && method === 'GET') return fulfill({ items: problems, total: problems.length })
    if (path.startsWith('/api/problems/') && path.endsWith('/versions') && method === 'GET') return fulfill({ items: state.versions, total: state.versions.length })
    if (path.startsWith('/api/problems/') && method === 'GET') return fulfill(problemDetail(decodeURIComponent(path.split('/').at(-1) ?? '')))

    if (path.startsWith('/api/drafts/') && method === 'GET') {
      const language = url.searchParams.get('language') ?? 'cuda_cpp'
      const draft = drafts.get(language) ?? ''
      return draft
        ? fulfill({ problem_id: 'vector-addition', language, source: draft, updated_at: '2026-08-16T01:00:00Z' })
        : fulfill({ detail: '尚未保存草稿' }, 404)
    }
    if (path.startsWith('/api/drafts/') && method === 'PUT') {
      const body = request.postDataJSON() as { language: string; source: string }
      drafts.set(body.language, body.source)
      return fulfill({ problem_id: 'vector-addition', language: body.language, source: body.source, updated_at: new Date().toISOString() })
    }

    if (path === '/api/versions/duplicates' && method === 'GET') return fulfill({ duplicate: false, items: [] })
    if (path === '/api/versions/compare' && method === 'POST') {
      const body = request.postDataJSON() as { version_ids: string[]; baseline_id: string }
      const baseline = state.versions.find((item) => item.id === body.baseline_id) ?? state.versions[0]
      const comparable = options.comparable !== false
      const rows = ['64K', '1M'].map((size, sizeIndex) => {
        const baselineMedian = baseline.benchmark_runs[0].measurements[sizeIndex].median_ms
        return {
          size,
          metrics: Object.fromEntries(body.version_ids.map((id) => {
            const version = state.versions.find((item) => item.id === id)!
            const metric = version.benchmark_runs[0].measurements[sizeIndex]
            return [id, { ...metric, sample_count: metric.samples_ms.length, speedup: comparable ? baselineMedian / metric.median_ms : null }]
          })),
        }
      })
      return fulfill({
        problem_id: 'vector-addition', baseline_id: body.baseline_id, comparable,
        environment_consistent: comparable, reasons: comparable ? [] : ['GPU/CUDA 环境指纹不同'], rows,
      })
    }
    if (path.startsWith('/api/versions/') && method === 'PATCH') {
      const id = path.split('/').at(-1)
      const changes = request.postDataJSON() as { name?: string; notes?: string }
      const version = state.versions.find((item) => item.id === id)
      if (!version) return fulfill({ detail: '版本不存在' }, 404)
      if (changes.name !== undefined) version.name = changes.name
      if (changes.notes !== undefined) version.notes = changes.notes
      return fulfill(version)
    }
    if (path.startsWith('/api/versions/') && method === 'DELETE') {
      state.deleteConfirmed = url.searchParams.get('confirmed') === 'true'
      if (!state.deleteConfirmed) return fulfill({ detail: '删除版本需要二次确认' }, 409)
      const id = path.split('/').at(-1)
      state.versions = state.versions.filter((item) => item.id !== id)
      return route.fulfill({ status: 204, body: '' })
    }

    if (path === '/api/jobs' && method === 'POST') {
      const body = request.postDataJSON() as Record<string, unknown>
      state.submittedJobs.push(body)
      const id = `job-${state.submittedJobs.length}`
      jobs.set(id, body)
      return fulfill({ id, problem_id: body.problem_id, language: body.language, action: body.action, status: 'queued', phase: 'queued', progress: 0 })
    }
    if (path.startsWith('/api/jobs/') && method === 'GET') {
      const id = path.split('/').at(-1) ?? ''
      const body = jobs.get(id) ?? {}
      const base = { id, problem_id: 'vector-addition', language: body.language, action: body.action, progress: 1 }
      if (body.action === 'compile') return fulfill({ ...base, status: 'failed', phase: 'compiling', error: { code: 'compile_error', message: body.language === 'triton_python' ? 'Triton 编译检查失败' : 'NVCC 编译失败' }, diagnostics: body.language === 'triton_python' ? 'solution.py:7: SyntaxError' : 'solution.cu:7: error: expected a semicolon' })
      if (body.action === 'run') return fulfill({ ...base, status: 'timed_out', phase: 'public', error: { code: 'timeout', message: '公开样例运行超过限制' } })
      if (body.action === 'validate') return fulfill({ ...base, status: 'failed', phase: 'full', error: { code: 'wrong_answer', message: '结果与参考实现不一致', details: { correctness: { cases: [{ name: '公开样例 1', passed: false, message: '第 17 个元素不匹配', error_type: 'wrong_answer' }] } } } })
      if (body.action === 'save_version') {
        if (!state.versions.some((item) => item.id === 'saved-v3')) {
          const created = savedVersion('saved-v3', String(body.version_name), 0.8, environment.fingerprint, body.language === 'triton_python' ? 'triton_python' : 'cuda_cpp')
          created.source_code = String(body.source)
          state.versions.push(created)
        }
        return fulfill({ ...base, status: 'succeeded', phase: 'completed', result: { version_id: 'saved-v3', message: '版本已保存' } })
      }
      return fulfill({ ...base, status: 'succeeded', phase: 'completed', result: { message: '统一环境重测完成' } })
    }

    return fulfill({ detail: `未模拟的接口：${method} ${path}` }, 404)
  })
  return state
}
