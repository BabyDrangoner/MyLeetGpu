import type { Job, JobResult } from '../../domain/types'
import { asDate, asKernelLanguage, asNumber, asRecord, asString } from './common'

export function normalizeJob(value: unknown): Job {
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
    problem_id: asString(raw.problem_id) || undefined,
    execution_target: raw.execution_target === 'cpu' ? 'cpu' : raw.execution_target === 'colab' ? 'colab' : raw.execution_target === 'local' ? 'local' : undefined,
    language: asKernelLanguage(raw.language),
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
