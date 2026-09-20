import { createDraftsClient } from './resources/drafts'
import { createExecutionClient } from './resources/execution'
import { createJobsClient } from './resources/jobs'
import { createProblemsClient } from './resources/problems'
import { createVersionsClient } from './resources/versions'
import { request, type HttpTransport } from './transport'

export { ApiError } from './transport'
export type { CreateJobInput } from './resources/jobs'

// This composition root is the stable entry point for application consumers.
export const createApiClient = (transport: HttpTransport = request) => ({
  problems: createProblemsClient(transport),
  drafts: createDraftsClient(transport),
  jobs: createJobsClient(transport),
  versions: createVersionsClient(transport),
  ...createExecutionClient(transport),
})

export const api = createApiClient()
