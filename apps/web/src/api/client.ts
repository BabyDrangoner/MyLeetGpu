import type {
  BenchmarkMetric,
  BenchmarkRun,
  ComparisonRow,
  ComparisonResult,
  Draft,
  EnvironmentSnapshot,
  Job,
  JobAction,
  JobResult,
  ProblemDetail,
  ProblemSummary,
  SavedVersion,
} from '../domain/types'

const API_ROOT = '/api'

export class ApiError extends Error {
  readonly status: number
  readonly details: unknown

  constructor(message: string, status: number, details?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.details = details
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), 15_000)
  try {
    const response = await fetch(`${API_ROOT}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
      signal: init?.signal ?? controller.signal,
    })
    const contentType = response.headers.get('content-type') ?? ''
    const payload: unknown = response.status === 204
      ? undefined
      : contentType.includes('application/json')
        ? await response.json()
        : await response.text()

    if (!response.ok) {
      const payloadRecord = asRecord(payload)
      const apiError = asRecord(payloadRecord.error)
      const detail = payloadRecord.detail ?? apiError.message ?? payload
      const message = typeof detail === 'string'
        ? detail
        : `请求失败（HTTP ${response.status}）`
      throw new ApiError(message, response.status, detail)
    }
    return payload as T
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError('连接 API 超时，请确认服务仍在运行。', 408)
    }
    throw new ApiError(error instanceof Error ? error.message : '无法连接本地 API。', 0)
  } finally {
    window.clearTimeout(timer)
  }
}

function listItems<T>(payload: { items?: T[] } | T[]): T[] {
  return Array.isArray(payload) ? payload : payload.items ?? []
}

type JsonRecord = Record<string, unknown>

function asRecord(value: unknown): JsonRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as JsonRecord : {}
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : value === null || value === undefined ? fallback : String(value)
}

function asNumber(value: unknown, fallback = 0): number {
  const parsed = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function asDate(value: unknown): string {
  const text = asString(value)
  if (!text) return ''
  return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?$/.test(text) ? `${text}Z` : text
}

function normalizeProblemSummary(value: unknown): ProblemSummary {
  const raw = asRecord(value)
  return {
    slug: asString(raw.slug ?? raw.id),
    title: asString(raw.title, '未命名题目'),
    difficulty: asString(raw.difficulty, 'easy'),
    revision: asString(raw.revision, '1'),
    summary: asString(raw.summary),
  }
}

function displayConstraint(name: string, value: unknown): string {
  const detail = asRecord(value)
  if ('min' in detail || 'max' in detail) {
    const bounds = [detail.min !== undefined ? `min=${detail.min}` : '', detail.max !== undefined ? `max=${detail.max}` : ''].filter(Boolean)
    return `${name}: ${bounds.join('，')}`
  }
  if (Object.keys(detail).length) {
    return `${name}: ${Object.entries(detail).map(([key, item]) => `${key}=${String(item)}`).join('，')}`
  }
  return `${name}: ${asString(value)}`
}

function normalizeProblemDetail(value: unknown): ProblemDetail {
  const raw = asRecord(value)
  const signature = raw.signature
  const signatureRecord = asRecord(signature)
  const constraints = Array.isArray(raw.constraints)
    ? raw.constraints.map((item) => asString(item))
    : Object.entries(asRecord(raw.constraints)).map(([name, item]) => displayConstraint(name, item))
  const benchmarkRaw = asRecord(raw.benchmark)
  const sizesRaw = Array.isArray(benchmarkRaw.input_sizes)
    ? benchmarkRaw.input_sizes
    : Array.isArray(benchmarkRaw.sizes)
      ? benchmarkRaw.sizes
      : []
  const inputSizes = sizesRaw.map((item) => {
    const itemRecord = asRecord(item)
    return itemRecord.label !== undefined ? asString(itemRecord.label) : typeof item === 'number' ? item : asString(item)
  })
  return {
    ...normalizeProblemSummary(raw),
    statement_markdown: asString(raw.statement_markdown),
    starter_code: asString(raw.starter_code),
    signature: typeof signature === 'string'
      ? signature
      : asString(signatureRecord.declaration ?? signatureRecord.symbol),
    constraints,
    benchmark: { ...benchmarkRaw, input_sizes: inputSizes },
  }
}

function normalizeEnvironment(value: unknown): EnvironmentSnapshot {
  const raw = asRecord(value)
  const telemetry = asRecord(raw.telemetry)
  const unavailable = Array.isArray(raw.unavailable_metrics)
    ? raw.unavailable_metrics.map((item) => asString(item))
    : Object.entries(telemetry).filter(([, item]) => item === null || item === undefined || item === '').map(([key]) => key)
  const healthy = typeof raw.healthy === 'boolean' ? raw.healthy : raw.status === 'healthy'
  return {
    ...raw,
    healthy,
    status: asString(raw.status, healthy ? 'healthy' : 'unhealthy'),
    gpu_name: asString(raw.gpu_name ?? raw.gpu) || undefined,
    compute_capability: asString(raw.compute_capability) || undefined,
    driver_version: asString(raw.driver_version) || undefined,
    cuda_runtime_version: asString(raw.cuda_runtime_version ?? raw.cuda_version) || undefined,
    nvcc_version: asString(raw.nvcc_version) || undefined,
    container_image: asString(raw.container_image ?? raw.cuda_image) || undefined,
    container_digest: asString(raw.container_digest ?? raw.image_digest) || undefined,
    fingerprint: asString(raw.fingerprint) || undefined,
    checked_at: asDate(raw.checked_at ?? raw.observed_at ?? raw.created_at) || undefined,
    unavailable_metrics: unavailable,
    message: asString(raw.message ?? raw.error) || undefined,
    telemetry: telemetry as Record<string, string | null>,
  }
}

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

function normalizeBenchmarkRun(value: unknown): BenchmarkRun {
  const raw = asRecord(value)
  const metricsRaw = [raw.metrics, raw.results, raw.measurements, raw.measurements_json].find(Array.isArray) as unknown[] | undefined
  const environment = normalizeEnvironment(raw.environment)
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

function normalizeVersion(value: unknown): SavedVersion {
  const raw = asRecord(value)
  const runs = Array.isArray(raw.benchmark_runs)
    ? raw.benchmark_runs.map(normalizeBenchmarkRun).sort((left, right) => {
        const timeOrder = (Date.parse(left.created_at ?? '') || 0) - (Date.parse(right.created_at ?? '') || 0)
        return timeOrder || String(left.id ?? '').localeCompare(String(right.id ?? ''), undefined, { numeric: true })
      })
    : []
  return {
    id: asString(raw.id),
    problem_id: asString(raw.problem_id),
    problem_revision: asString(raw.problem_revision),
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

function normalizeJob(value: unknown): Job {
  const raw = asRecord(value)
  const resultRaw = asRecord(raw.result ?? raw.result_json)
  const correctness = asRecord(resultRaw.correctness)
  const errorRaw = raw.error ?? raw.error_json
  const error = asRecord(errorRaw)
  const errorCorrectness = asRecord(asRecord(error.details).correctness)
  const cases = [resultRaw.test_cases, resultRaw.cases, correctness.cases, errorCorrectness.cases].find(Array.isArray)
  const progressValue = raw.progress === undefined ? undefined : asNumber(raw.progress)
  const result: JobResult | null = Object.keys(resultRaw).length || Array.isArray(cases) ? {
    ...resultRaw,
    test_cases: Array.isArray(cases) ? cases : undefined,
    stdout: asString(resultRaw.stdout ?? resultRaw.output) || undefined,
    summary: asString(resultRaw.summary ?? resultRaw.message) || undefined,
  } : null
  return {
    id: asString(raw.id),
    status: asString(raw.status, 'queued') as Job['status'],
    action: asString(raw.action, 'compile') as Job['action'],
    phase: asString(raw.phase) || undefined,
    progress: progressValue === undefined ? undefined : progressValue <= 1 ? progressValue * 100 : progressValue,
    queue_position: raw.queue_position === undefined ? undefined : asNumber(raw.queue_position),
    created_at: asDate(raw.created_at) || undefined,
    updated_at: asDate(raw.updated_at ?? raw.completed_at ?? raw.started_at) || undefined,
    result: result as Job['result'],
    error: typeof errorRaw === 'string' ? errorRaw : Object.keys(error).length ? error : null,
    diagnostics: asString(raw.diagnostics ?? resultRaw.compile_diagnostics) || null,
  }
}

function normalizeComparison(value: unknown, versionIds: string[], baselineId: string): ComparisonResult {
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
    version_ids: Array.isArray(raw.version_ids) ? raw.version_ids.map((id) => asString(id)) : versionIds,
    environment: raw.environment ? normalizeEnvironment(raw.environment) : undefined,
  }
}

export interface CreateJobInput {
  problem_id: string
  action: JobAction
  source?: string
  version_name?: string
  notes?: string
  version_ids?: string[]
  allow_duplicate?: boolean
}

export const api = {
  problems: {
    async list(): Promise<ProblemSummary[]> {
      const payload = await request<{ items?: unknown[] } | unknown[]>('/problems')
      return listItems(payload).map(normalizeProblemSummary)
    },
    async get(slug: string): Promise<ProblemDetail> {
      return normalizeProblemDetail(await request(`/problems/${encodeURIComponent(slug)}`))
    },
  },

  drafts: {
    async get(problemId: string): Promise<Draft> {
      const raw = asRecord(await request(`/drafts/${encodeURIComponent(problemId)}`))
      return { problem_id: asString(raw.problem_id, problemId), source: asString(raw.source ?? raw.source_code), updated_at: asDate(raw.updated_at) }
    },
    async save(problemId: string, source: string): Promise<Draft> {
      const raw = asRecord(await request(`/drafts/${encodeURIComponent(problemId)}`, {
        method: 'PUT',
        body: JSON.stringify({ problem_id: problemId, source }),
      }))
      return { problem_id: asString(raw.problem_id, problemId), source: asString(raw.source ?? raw.source_code), updated_at: asDate(raw.updated_at) }
    },
  },

  jobs: {
    async create(input: CreateJobInput): Promise<Job> {
      return normalizeJob(await request('/jobs', { method: 'POST', body: JSON.stringify(input) }))
    },
    async get(id: string): Promise<Job> {
      return normalizeJob(await request(`/jobs/${encodeURIComponent(id)}`))
    },
  },

  versions: {
    async list(problemId: string): Promise<SavedVersion[]> {
      const payload = await request<{ items?: unknown[] } | unknown[]>(
        `/problems/${encodeURIComponent(problemId)}/versions`,
      )
      return listItems(payload).map(normalizeVersion)
    },
    async update(id: string, changes: Pick<SavedVersion, 'name' | 'notes'>): Promise<SavedVersion> {
      return normalizeVersion(await request(`/versions/${encodeURIComponent(id)}`, {
        method: 'PATCH',
        body: JSON.stringify(changes),
      }))
    },
    remove(id: string): Promise<void> {
      return request(`/versions/${encodeURIComponent(id)}?confirmed=true`, { method: 'DELETE' })
    },
    async compare(problemId: string, versionIds: string[], baselineId: string): Promise<ComparisonResult> {
      const result = await request('/versions/compare', {
        method: 'POST',
        body: JSON.stringify({
          problem_id: problemId,
          version_ids: versionIds,
          baseline_id: baselineId,
        }),
      })
      return normalizeComparison(result, versionIds, baselineId)
    },
    async findDuplicates(problemId: string, sourceHash: string): Promise<SavedVersion[]> {
      const query = new URLSearchParams({ problem_id: problemId, source_hash: sourceHash })
      try {
        const payload = await request<{ items?: unknown[] } | unknown[]>(`/versions/duplicates?${query}`)
        return listItems(payload).map(normalizeVersion)
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return []
        throw error
      }
    },
  },

  async environment(): Promise<EnvironmentSnapshot> {
    return normalizeEnvironment(await request('/environment'))
  },
}
