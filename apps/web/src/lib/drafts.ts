export interface LocalDraft {
  source: string
  updatedAt: string
}

const draftKey = (problemId: string) => `myleetgpu:draft:${problemId}`

export function readLocalDraft(problemId: string): LocalDraft | null {
  try {
    const raw = localStorage.getItem(draftKey(problemId))
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<LocalDraft>
    return typeof parsed.source === 'string' && typeof parsed.updatedAt === 'string'
      ? { source: parsed.source, updatedAt: parsed.updatedAt }
      : null
  } catch {
    return null
  }
}

export function saveLocalDraft(problemId: string, source: string): LocalDraft {
  const draft = { source, updatedAt: new Date().toISOString() }
  try {
    localStorage.setItem(draftKey(problemId), JSON.stringify(draft))
  } catch {
    // The editor remains usable when browser storage is unavailable.
  }
  return draft
}
