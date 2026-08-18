import type { BenchmarkMetric, BenchmarkRun, ComparisonRow, SavedVersion } from '../domain/types'

export function latestBenchmarkRun(version: SavedVersion): BenchmarkRun | undefined {
  return [...(version.benchmark_runs ?? [])].sort((left, right) => {
    const byTime = (left.created_at ?? '').localeCompare(right.created_at ?? '')
    return byTime || (left.id ?? '').localeCompare(right.id ?? '')
  }).at(-1)
}

export function latestMetrics(version: SavedVersion): BenchmarkMetric[] {
  const latest = latestBenchmarkRun(version)
  return latest?.metrics ?? latest?.results ?? latest?.measurements ?? []
}

export function comparisonMetric(row: ComparisonRow, versionId: string): BenchmarkMetric | undefined {
  const fromList = row.versions?.find((metric) => metric.version_id === versionId)
  if (fromList) return fromList
  const metric = row.metrics?.[versionId]
  return metric ? { ...metric, size: metric.size ?? row.size } : undefined
}

export function runFingerprint(run?: BenchmarkRun): string | undefined {
  return run?.environment_fingerprint ?? run?.environment?.fingerprint
}

export function localComparability(versions: SavedVersion[]): { comparable: boolean; reasons: string[] } {
  if (versions.length < 2) return { comparable: false, reasons: ['至少选择两个版本'] }
  const values = <T,>(selector: (version: SavedVersion) => T | undefined) =>
    new Set(versions.map(selector).filter((value) => value !== undefined))
  const reasons: string[] = []
  if (values((version) => version.language).size > 1) reasons.push('实现语言不一致')
  if (values((version) => version.problem_revision).size > 1) reasons.push('题目修订版本不一致')
  if (values((version) => latestBenchmarkRun(version)?.suite_hash).size > 1) reasons.push('测试套件不一致')
  if (values((version) => {
    const latest = latestBenchmarkRun(version)
    return JSON.stringify(latest?.compiler_flags ?? latest?.compile_flags ?? version.compile_flags)
  }).size > 1) {
    reasons.push('编译配置不一致')
  }
  if (values((version) => runFingerprint(latestBenchmarkRun(version))).size > 1) reasons.push('运行环境指纹不一致')
  const sizes = versions.map((version) => latestMetrics(version).map((metric) => String(metric.size)).sort().join('|'))
  if (new Set(sizes).size > 1) reasons.push('输入规模不一致')
  return { comparable: reasons.length === 0, reasons }
}
