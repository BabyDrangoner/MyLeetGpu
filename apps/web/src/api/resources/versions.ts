import type { ComparisonResult, KernelLanguage, SavedVersion } from '../../domain/types'
import { listItems } from '../mappers/common'
import { normalizeComparison, normalizeVersion } from '../mappers/versions'
import { ApiError, type HttpTransport } from '../transport'

export const createVersionsClient = (request: HttpTransport) => ({
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
  async compare(problemId: string, language: KernelLanguage, versionIds: string[], baselineId: string): Promise<ComparisonResult> {
    const result = await request('/versions/compare', {
      method: 'POST',
      body: JSON.stringify({
        problem_id: problemId,
        language,
        version_ids: versionIds,
        baseline_id: baselineId,
      }),
    })
    return normalizeComparison(result, versionIds, baselineId, language)
  },
  async findDuplicates(problemId: string, language: KernelLanguage, sourceHash: string): Promise<SavedVersion[]> {
    const query = new URLSearchParams({ problem_id: problemId, language, source_hash: sourceHash })
    try {
      const payload = await request<{ items?: unknown[] } | unknown[]>(`/versions/duplicates?${query}`)
      return listItems(payload).map(normalizeVersion)
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) return []
      throw error
    }
  },
})
