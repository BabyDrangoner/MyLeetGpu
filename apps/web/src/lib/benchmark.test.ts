import { describe, expect, it } from 'vitest'
import type { SavedVersion } from '../domain/types'
import { comparisonMetric, latestBenchmarkRun, localComparability } from './benchmark'

function version(id: string, overrides: Partial<SavedVersion> = {}): SavedVersion {
  return {
    id,
    problem_id: 'vector-add',
    problem_revision: '1',
    name: id,
    source_hash: id,
    source_code: '// code',
    created_at: '2026-01-01T00:00:00Z',
    correctness_status: 'passed',
    benchmark_runs: [{
      suite_hash: 'suite-a',
      compiler_flags: ['-O3', '-arch=sm_89'],
      environment_fingerprint: 'env-a',
      metrics: [{ size: 1024, median_ms: 0.2, p95_ms: 0.22, sample_count: 30 }],
    }],
    ...overrides,
  }
}

describe('benchmark comparability', () => {
  it('accepts versions measured with the same complete context', () => {
    expect(localComparability([version('a'), version('b')])).toEqual({ comparable: true, reasons: [] })
  })

  it('reports every mismatched dimension', () => {
    const changed = version('b', {
      problem_revision: '2',
      benchmark_runs: [{
        suite_hash: 'suite-b',
        compiler_flags: ['-O2'],
        environment_fingerprint: 'env-b',
        metrics: [{ size: 2048, median_ms: 0.3, p95_ms: 0.34, sample_count: 30 }],
      }],
    })
    const result = localComparability([version('a'), changed])
    expect(result.comparable).toBe(false)
    expect(result.reasons).toEqual(expect.arrayContaining([
      '题目修订版本不一致',
      '测试套件不一致',
      '编译配置不一致',
      '运行环境指纹不一致',
      '输入规模不一致',
    ]))
  })

  it('reads both supported comparison row shapes', () => {
    expect(comparisonMetric({
      size: 4096,
      versions: [{ version_id: 'a', size: 4096, median_ms: 0.4, p95_ms: 0.5, sample_count: 20, speedup: 1.2 }],
    }, 'a')?.speedup).toBe(1.2)
    expect(comparisonMetric({
      size: 4096,
      metrics: { a: { size: 4096, median_ms: 0.4, p95_ms: 0.5, sample_count: 20, speedup: 1.2 } },
    }, 'a')?.median_ms).toBe(0.4)
  })

  it('selects the newest run even when the API relation is unordered', () => {
    const saved = version('a', {
      benchmark_runs: [
        { id: 'new', created_at: '2026-02-01T00:00:00Z', suite_hash: 'suite-new' },
        { id: 'old', created_at: '2026-01-01T00:00:00Z', suite_hash: 'suite-old' },
      ],
    })

    expect(latestBenchmarkRun(saved)?.id).toBe('new')
  })
})
