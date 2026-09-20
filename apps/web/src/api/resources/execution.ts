import type { EnvironmentSnapshot, ExecutionSettings, ExecutionTarget, KernelLanguage } from '../../domain/types'
import { normalizeEnvironment } from '../mappers/environment'
import type { HttpTransport } from '../transport'

export const createExecutionClient = (request: HttpTransport) => ({
  executionSettings: {
    get(): Promise<ExecutionSettings> {
      return request('/execution-settings')
    },
    save(input: Pick<ExecutionSettings, 'target' | 'colab_acknowledged'>): Promise<ExecutionSettings> {
      return request('/execution-settings', { method: 'PUT', body: JSON.stringify(input) })
    },
    async probe(target: ExecutionTarget, language: KernelLanguage): Promise<EnvironmentSnapshot> {
      // A queued GPU probe includes waiting for the serial worker, not just HTTP latency.
      return normalizeEnvironment(await request('/execution-settings/probe', {
        method: 'POST', body: JSON.stringify({ target, language }),
      }, 195_000), language)
    },
  },
  async environment(language: KernelLanguage = 'cuda_cpp'): Promise<EnvironmentSnapshot> {
    const query = new URLSearchParams({ language })
    return normalizeEnvironment(await request(`/environment?${query}`), language)
  },
})
