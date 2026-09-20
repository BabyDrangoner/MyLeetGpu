import type { BenchmarkMetric, BenchmarkRun, ComparisonResult, ComparisonRow, KernelLanguage, SavedVersion } from '../../domain/types'
import { asDate, asKernelLanguage, asNumber, asRecord, asString } from './common'
import { normalizeEnvironment } from './environment'

function normalizeMetric(value: unknown, fallbackSize: string | number = ''): BenchmarkMetric {
  const raw = asRecord(value)
  const samples = Array.isArray(raw.samples_ms) ? raw.samples_ms.map((item) => asNumber(item)) : undefined
  return {
    ...raw,
    size: typeof raw.size === 'number' ? raw.size : asString(raw.size, asString(fallbackSize)),
    median_ms: asNumber(raw.median_ms),
    p95_ms: asNumber(raw.p95_ms),
    min_ms: raw.min_ms === undefined || raw.min_ms === null ? undefined : asNumber(raw.min_ms),
    cv: raw.cv === undefined || raw.cv === null ? undefined : asNumber(raw.cv),
    mad_ms: raw.mad_ms === undefined || raw.mad_ms === null ? undefined : asNumber(raw.mad_ms),
    sample_count: asNumber(raw.sample_count, samples?.length ?? 0),
    samples_ms: samples,
    inner_repetitions: raw.inner_repetitions === undefined ? undefined : asNumber(raw.inner_repetitions),
    speedup: raw.speedup === undefined || raw.speedup === null ? null : asNumber(raw.speedup),
  }
}

function normalizeBenchmarkRun(value: unknown, language: KernelLanguage = 'cuda_cpp'): BenchmarkRun {
  const raw = asRecord(value)
  const metricsRaw = [raw.metrics, raw.results, raw.measurements, raw.measurements_json].find(Array.isArray) as unknown[] | undefined
  const environment = normalizeEnvironment(raw.environment, language)
  return {
    ...raw,
    id: asString(raw.id) || undefined,
    version_id: asString(raw.version_id) || undefined,
    suite_hash: asString(raw.suite_hash) || undefined,
    protocol_version: asString(raw.protocol_version) || undefined,
    compiler_flags: (raw.compiler_flags ?? raw.compile_flags ?? raw.compile_flags_json) as string[] | string | undefined,
    compile_flags: (raw.compile_flags ?? raw.compiler_flags ?? raw.compile_flags_json) as string[] | string | undefined,
    random_seed: raw.random_seed === undefined && raw.seed === undefined ? undefined : asNumber(raw.random_seed ?? raw.seed),
    seed: raw.seed === undefined && raw.random_seed === undefined ? undefined : asNumber(raw.seed ?? raw.random_seed),
    warmup: raw.warmup === undefined ? undefined : asNumber(raw.warmup),
    iterations: raw.iterations === undefined ? undefined : asNumber(raw.iterations),
    environment_fingerprint: asString(raw.environment_fingerprint ?? environment.fingerprint) || undefined,
    environment,
    metrics: (metricsRaw ?? []).map((metric) => normalizeMetric(metric)),
    measurements: (metricsRaw ?? []).map((metric) => normalizeMetric(metric)),
    created_at: asDate(raw.created_at) || undefined,
  }
}

export function normalizeVersion(value: unknown): SavedVersion {
  const raw = asRecord(value)
  const language = asKernelLanguage(raw.language)
  const runs = Array.isArray(raw.benchmark_runs)
    ? raw.benchmark_runs.map((run) => normalizeBenchmarkRun(run, language)).sort((left, right) => {
        const timeOrder = (Date.parse(left.created_at ?? '') || 0) - (Date.parse(right.created_at ?? '') || 0)
        return timeOrder || String(left.id ?? '').localeCompare(String(right.id ?? ''), undefined, { numeric: true })
      })
    : []
  return {
    id: asString(raw.id),
    problem_id: asString(raw.problem_id),
    problem_revision: asString(raw.problem_revision),
    language,
    name: asString(raw.name, '未命名版本'),
    notes: asString(raw.notes) || undefined,
    source_hash: asString(raw.source_hash),
    source_code: asString(raw.source_code),
    created_at: asDate(raw.created_at),
    correctness_status: asString(raw.correctness_status, 'unknown'),
    benchmark_runs: runs,
    compile_flags: (raw.compile_flags ?? raw.compile_flags_json) as string[] | string | undefined,
  }
}

export function normalizeComparison(value: unknown, versionIds: string[], baselineId: string, fallbackLanguage: KernelLanguage): ComparisonResult {
  const raw = asRecord(value)
  const rows: ComparisonRow[] = (Array.isArray(raw.rows) ? raw.rows : []).map((item) => {
    const row = asRecord(item)
    const metricsRaw = asRecord(row.metrics)
    const metrics = Object.fromEntries(Object.entries(metricsRaw).map(([id, metric]) => [
      id,
      metric === null ? null : normalizeMetric(metric, asString(row.size)),
    ]))
    const versions = Array.isArray(row.versions)
      ? row.versions.map((metric) => {
        const normalized = normalizeMetric(metric, asString(row.size))
        return { ...normalized, version_id: asString(asRecord(metric).version_id) }
      })
      : undefined
    return { size: typeof row.size === 'number' ? row.size : asString(row.size), metrics, versions }
  })
  return {
    comparable: Boolean(raw.comparable),
    reasons: Array.isArray(raw.reasons) ? raw.reasons.map((reason) => asString(reason)) : [],
    environment_consistent: Boolean(raw.environment_consistent),
    rows,
    baseline_id: asString(raw.baseline_id, baselineId),
    language: asKernelLanguage(raw.language, fallbackLanguage),
    version_ids: Array.isArray(raw.version_ids) ? raw.version_ids.map((id) => asString(id)) : versionIds,
    environment: raw.environment ? normalizeEnvironment(raw.environment, fallbackLanguage) : undefined,
  }
}
