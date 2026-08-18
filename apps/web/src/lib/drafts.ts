import type { KernelLanguage } from '../domain/types'

export interface LocalDraft {
  language: KernelLanguage
  source: string
  updatedAt: string
}

const draftKey = (problemId: string, language: KernelLanguage) => `myleetgpu:draft:${problemId}:${language}`
const legacyDraftKey = (problemId: string) => `myleetgpu:draft:${problemId}`

function parseDraft(raw: string | null, language: KernelLanguage): LocalDraft | null {
  if (!raw) return null
  const parsed = JSON.parse(raw) as Partial<LocalDraft>
  return typeof parsed.source === 'string' && typeof parsed.updatedAt === 'string'
    ? { language, source: parsed.source, updatedAt: parsed.updatedAt }
    : null
}

export function readLocalDraft(problemId: string, language: KernelLanguage): LocalDraft | null {
  try {
    const current = parseDraft(localStorage.getItem(draftKey(problemId, language)), language)
    if (current || language !== 'cuda_cpp') return current

    const legacy = parseDraft(localStorage.getItem(legacyDraftKey(problemId)), language)
    if (!legacy) return null
    localStorage.setItem(draftKey(problemId, language), JSON.stringify(legacy))
    localStorage.removeItem(legacyDraftKey(problemId))
    return legacy
  } catch {
    return null
  }
}

export function saveLocalDraft(problemId: string, language: KernelLanguage, source: string): LocalDraft {
  const draft = { language, source, updatedAt: new Date().toISOString() }
  try {
    localStorage.setItem(draftKey(problemId, language), JSON.stringify(draft))
  } catch {
    // The editor remains usable when browser storage is unavailable.
  }
  return draft
}
