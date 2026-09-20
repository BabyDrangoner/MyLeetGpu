import type { ProblemDetail, ProblemSummary } from '../../domain/types'
import { listItems } from '../mappers/common'
import { normalizeProblemDetail, normalizeProblemSummary } from '../mappers/problems'
import type { HttpTransport } from '../transport'

export const createProblemsClient = (request: HttpTransport) => ({
  async list(): Promise<ProblemSummary[]> {
    const payload = await request<{ items?: unknown[] } | unknown[]>('/problems')
    return listItems(payload).map(normalizeProblemSummary)
  },
  async get(slug: string): Promise<ProblemDetail> {
    return normalizeProblemDetail(await request(`/problems/${encodeURIComponent(slug)}`))
  },
})
