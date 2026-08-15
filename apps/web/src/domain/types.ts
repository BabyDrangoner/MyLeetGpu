export type Difficulty = '入门' | '简单' | '中等' | '困难' | string

export interface ProblemSummary {
  slug: string
  title: string
  difficulty: Difficulty
  revision: string
  summary: string
}

export interface BenchmarkProtocol {
  input_sizes?: Array<string | number>
  warmup?: number
  iterations?: number
  timeout_seconds?: number
  suite_hash?: string
  protocol_version?: string
  [key: string]: unknown
}

export interface ProblemDetail extends ProblemSummary {
  statement_markdown: string
  starter_code: string
  signature?: string
  constraints?: string[]
  benchmark?: BenchmarkProtocol
}

export interface Draft {
  problem_id: string
  source: string
  updated_at: string
}

export type JobAction = 'compile' | 'run' | 'validate' | 'save_version' | 'rebenchmark'
export type JobStatus =
  | 'queued'
  | 'compiling'
  | 'running'
  | 'validating'
  | 'benchmarking'
  | 'succeeded'
  | 'failed'
  | 'timed_out'
  | 'cancelled'
  | 'system_error'

export interface TestCaseResult {
  name?: string
  passed?: boolean
  status?: string
  duration_ms?: number
  message?: string
  error_type?: string
  stdout?: string
  stderr?: string
}

export interface JobResult {
  message?: string
  version_id?: string
  compile_succeeded?: boolean
  passed?: boolean
  test_cases?: TestCaseResult[]
  cases?: TestCaseResult[]
  stdout?: string
  stderr?: string
  summary?: string
  benchmark?: BenchmarkRun
  [key: string]: unknown
}

export interface JobError {
  code?: string
  message?: string
  category?: string
  stage?: string
  details?: unknown
}

export interface Job {
  id: string
  status: JobStatus
  action: JobAction
  phase?: string
  progress?: number
  queue_position?: number
  created_at?: string
  updated_at?: string
  result?: JobResult | null
  error?: JobError | string | null
  diagnostics?: string | null
}

export interface BenchmarkMetric {
  size: string | number
  median_ms: number
  p95_ms: number
  min_ms?: number
  cv?: number
  mad_ms?: number
  sample_count: number
  samples_ms?: number[]
  inner_repetitions?: number
  speedup?: number | null
}

export interface EnvironmentSnapshot {
  id?: string
  healthy?: boolean
  status?: string
  gpu_name?: string
  gpu?: string
  compute_capability?: string
  driver_version?: string
  cuda_runtime_version?: string
  cuda_version?: string
  nvcc_version?: string
  container_image?: string
  container_digest?: string
  cuda_image?: string
  image_digest?: string
  fingerprint?: string
  checked_at?: string
  unavailable_metrics?: string[]
  message?: string
  error?: string
  telemetry?: Record<string, string | null>
  [key: string]: unknown
}

export interface BenchmarkRun {
  id?: string
  version_id?: string
  suite_hash?: string
  protocol_version?: string
  compiler_flags?: string[] | string
  compile_flags?: string[] | string
  random_seed?: number
  seed?: number
  warmup?: number
  iterations?: number
  environment_fingerprint?: string
  environment?: EnvironmentSnapshot
  metrics?: BenchmarkMetric[]
  results?: BenchmarkMetric[]
  measurements?: BenchmarkMetric[]
  created_at?: string
  [key: string]: unknown
}

export interface SavedVersion {
  id: string
  problem_id: string
  problem_revision: string
  name: string
  notes?: string
  source_hash: string
  source_code: string
  created_at: string
  correctness_status: string
  benchmark_runs: BenchmarkRun[]
  compile_flags?: string[] | string
}

export interface ComparisonVersionMetric extends BenchmarkMetric {
  version_id: string
  version_name?: string
}

export interface ComparisonRow {
  size: string | number
  versions?: ComparisonVersionMetric[]
  metrics?: Record<string, Omit<ComparisonVersionMetric, 'version_id'> | null>
}

export interface ComparisonResult {
  comparable: boolean
  reasons: string[]
  environment_consistent: boolean
  rows: ComparisonRow[]
  baseline_id: string
  version_ids?: string[]
  environment?: EnvironmentSnapshot
}
