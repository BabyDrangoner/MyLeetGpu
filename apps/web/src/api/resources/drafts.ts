import type { Draft, KernelLanguage } from '../../domain/types'
import { normalizeDraft } from '../mappers/drafts'
import type { HttpTransport } from '../transport'

export const createDraftsClient = (request: HttpTransport) => ({
  async get(problemId: string, language: KernelLanguage): Promise<Draft> {
    const query = new URLSearchParams({ language })
    return normalizeDraft(await request(`/drafts/${encodeURIComponent(problemId)}?${query}`), problemId, language)
  },
  async save(problemId: string, language: KernelLanguage, source: string): Promise<Draft> {
    return normalizeDraft(await request(`/drafts/${encodeURIComponent(problemId)}`, {
      method: 'PUT',
      body: JSON.stringify({ language, source }),
    }), problemId, language)
  },
})
