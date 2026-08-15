import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, api } from './client'

const fetchMock = vi.fn()

function json(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json' } })
}

describe('API contract adapter', () => {
  beforeEach(() => {
    fetchMock.mockReset()
    vi.stubGlobal('fetch', fetchMock)
  })

  it('adapts the declarative problem manifest into the editor view model', async () => {
    fetchMock.mockResolvedValue(json({
      slug: 'vector-addition',
      title: '向量逐元素相加',
      difficulty: 'easy',
      revision: '1',
      summary: '原创向量题',
      statement_markdown: '# 题目',
      starter_code: 'void solve() {}',
      signature: { symbol: 'solve', declaration: 'void solve(float* output)' },
      constraints: { n: { min: 1, max: 1024 }, aliasing: 'none' },
      benchmark: { sizes: [{ label: '64K' }, { label: '1M' }], warmup: 8, iterations: 20 },
    }))
    const problem = await api.problems.get('vector-addition')
    expect(problem.signature).toBe('void solve(float* output)')
    expect(problem.constraints).toEqual(['n: min=1，max=1024', 'aliasing: none'])
    expect(problem.benchmark?.input_sizes).toEqual(['64K', '1M'])
  })

  it('normalizes jobs, environment snapshots and backend UTC timestamps', async () => {
    fetchMock
      .mockResolvedValueOnce(json({
        id: 'job-1', action: 'run', status: 'running', phase: 'public', progress: 0.45,
        created_at: '2026-08-16T01:00:00',
        result: { correctness: { status: 'passed', cases: [{ name: 'sample_1', passed: true }] }, output: 'ok' },
      }))
      .mockResolvedValueOnce(json({
        healthy: true, status: 'healthy', gpu_name: 'RTX 4060', cuda_image: 'cuda:12.6',
        image_digest: 'sha256:abc', observed_at: '2026-08-16T01:00:00',
        telemetry: { temperature_c: null, power_w: '95' },
      }))

    const job = await api.jobs.get('job-1')
    expect(job.progress).toBe(45)
    expect(job.result?.test_cases?.[0].name).toBe('sample_1')
    expect(job.result?.stdout).toBe('ok')
    expect(job.created_at).toBe('2026-08-16T01:00:00Z')

    const environment = await api.environment()
    expect(environment.container_image).toBe('cuda:12.6')
    expect(environment.container_digest).toBe('sha256:abc')
    expect(environment.unavailable_metrics).toEqual(['temperature_c'])
    expect(environment.checked_at).toBe('2026-08-16T01:00:00Z')
  })

  it('sends the server-required second-confirmation flag when deleting', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }))
    await api.versions.remove('version-1')
    expect(fetchMock).toHaveBeenCalledWith('/api/versions/version-1?confirmed=true', expect.objectContaining({ method: 'DELETE' }))
  })

  it('orders benchmark runs by creation time and id before selecting the latest run', async () => {
    fetchMock.mockResolvedValue(json({ items: [{
      id: 'version-1', problem_id: 'vector-addition', problem_revision: '1', name: 'v1',
      source_hash: 'abc', source_code: 'void solve() {}', created_at: '2026-08-16T01:00:00Z',
      correctness_status: 'passed',
      benchmark_runs: [
        { id: '12', created_at: '2026-08-16T02:00:00Z', protocol_version: 'latest' },
        { id: '2', created_at: '2026-08-16T01:00:00Z', protocol_version: 'old' },
      ],
    }] }))

    const versions = await api.versions.list('vector-addition')
    expect(versions[0].benchmark_runs.at(-1)?.protocol_version).toBe('latest')
  })

  it('surfaces structured API error messages', async () => {
    fetchMock.mockResolvedValue(json({ error: { code: 'invalid_request', message: '版本名称必填' } }, 400))
    await expect(api.jobs.create({ problem_id: 'vector-addition', action: 'save_version' })).rejects.toEqual(
      expect.objectContaining({ message: '版本名称必填', status: 400 } satisfies Partial<ApiError>),
    )
  })
})
