import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from './client'

const fetchMock = vi.fn()
const json = (body: unknown) => new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' } })

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('execution API contracts', () => {
  it('loads settings and saves only target and acknowledgement', async () => {
    const settings = { target: 'local', colab_acknowledged: false, colab: { ssh_host: 'colab-vscode', remote_root: '/content/project/myleetgpu-runner', isolation: 'trusted-native' }, updated_at: null }
    fetchMock.mockResolvedValueOnce(json(settings)).mockResolvedValueOnce(json({ ...settings, target: 'colab', colab_acknowledged: true }))
    expect(await api.executionSettings.get()).toEqual(settings)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/execution-settings')
    await api.executionSettings.save({ target: 'colab', colab_acknowledged: true })
    expect(fetchMock.mock.calls[1][1]).toMatchObject({ method: 'PUT', body: JSON.stringify({ target: 'colab', colab_acknowledged: true }) })
  })

  it('allows a queued probe to outlive the normal request timeout and normalizes its target', async () => {
    const timeout = vi.spyOn(window, 'setTimeout')
    fetchMock.mockResolvedValueOnce(json({ execution_target: 'colab', backend: 'triton_python', status: 'healthy', gpu: 'Tesla T4', toolchain: { torch_version: '2.5.1' } }))
    const result = await api.executionSettings.probe('colab', 'triton_python')
    expect(fetchMock.mock.calls[0][0]).toBe('/api/execution-settings/probe')
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: 'POST', body: JSON.stringify({ target: 'colab', language: 'triton_python' }) })
    expect(timeout).toHaveBeenCalledWith(expect.any(Function), 195_000)
    expect(result).toMatchObject({ execution_target: 'colab', backend: 'triton_python', healthy: true, gpu_name: 'Tesla T4', torch_version: '2.5.1' })
  })

  it('keeps a job execution-target snapshot distinct from the environment target', async () => {
    fetchMock.mockResolvedValueOnce(json({ id: 'old-job', execution_target: 'local', language: 'cuda_cpp', action: 'run', status: 'running' }))
      .mockResolvedValueOnce(json({ execution_target: 'colab', backend: 'cuda_cpp', healthy: true }))
    expect((await api.jobs.get('old-job')).execution_target).toBe('local')
    expect((await api.environment()).execution_target).toBe('colab')
  })

  it('preserves CPU execution and toolchain metadata without treating it as local GPU', async () => {
    fetchMock.mockResolvedValueOnce(json({
      execution_target: 'cpu', backend: 'cpp', healthy: true,
      toolchain: { cpu_name: 'Apple CPU', platform: 'Darwin', architecture: 'arm64', compiler_version: 'Apple clang 17' },
    })).mockResolvedValueOnce(json({ id: 'cpu-job', execution_target: 'cpu', language: 'python', action: 'run', status: 'succeeded' }))

    expect(await api.executionSettings.probe('cpu', 'cpp')).toMatchObject({
      execution_target: 'cpu', backend: 'cpp', cpu_name: 'Apple CPU', platform: 'Darwin', architecture: 'arm64', compiler_version: 'Apple clang 17',
    })
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ body: JSON.stringify({ target: 'cpu', language: 'cpp' }) })
    expect(await api.jobs.get('cpu-job')).toMatchObject({ execution_target: 'cpu', language: 'python' })
  })
})
