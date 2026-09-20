import type { Job, JobAction, KernelLanguage } from '../../domain/types'
import { normalizeJob } from '../mappers/jobs'
import type { HttpTransport } from '../transport'

export interface CreateJobInput {
  problem_id: string
  language: KernelLanguage
  action: JobAction
  source?: string
  version_name?: string
  notes?: string
  version_ids?: string[]
  allow_duplicate?: boolean
}

export const createJobsClient = (request: HttpTransport) => ({
  async create(input: CreateJobInput): Promise<Job> {
    return normalizeJob(await request('/jobs', { method: 'POST', body: JSON.stringify(input) }))
  },
  async get(id: string): Promise<Job> {
    return normalizeJob(await request(`/jobs/${encodeURIComponent(id)}`))
  },
})
