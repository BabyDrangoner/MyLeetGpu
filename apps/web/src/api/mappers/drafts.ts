import type { Draft, KernelLanguage } from '../../domain/types'
import { asDate, asKernelLanguage, asRecord, asString } from './common'

export function normalizeDraft(value: unknown, problemId: string, language: KernelLanguage): Draft {
  const raw = asRecord(value)
  return {
    problem_id: asString(raw.problem_id, problemId),
    language: asKernelLanguage(raw.language, language),
    source: asString(raw.source ?? raw.source_code),
    updated_at: asDate(raw.updated_at),
  }
}
