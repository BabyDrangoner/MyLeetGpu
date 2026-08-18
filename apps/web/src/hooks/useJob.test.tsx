import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { useJob } from './useJob'

vi.mock('../api/client', () => ({
  api: {
    jobs: {
      create: vi.fn(),
      get: vi.fn(),
    },
  },
}))

const createMock = vi.mocked(api.jobs.create)
const getMock = vi.mocked(api.jobs.get)

describe('useJob', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    createMock.mockReset()
    getMock.mockReset()
  })
  afterEach(() => vi.useRealTimers())

  it('polls an asynchronous job through phases to a terminal result', async () => {
    createMock.mockResolvedValue({ id: 'job-1', language: 'triton_python', action: 'validate', status: 'queued' })
    getMock
      .mockResolvedValueOnce({ id: 'job-1', language: 'triton_python', action: 'validate', status: 'validating', progress: 45 })
      .mockResolvedValueOnce({ id: 'job-1', language: 'triton_python', action: 'validate', status: 'succeeded', progress: 100, result: { message: '全部通过' } })
    const settled = vi.fn()
    const { result } = renderHook(() => useJob(settled))

    await act(async () => {
      await result.current.start({ problem_id: 'vector-addition', language: 'triton_python', action: 'validate', source: '# source' })
    })
    expect(result.current.job?.status).toBe('queued')

    await act(async () => { await vi.advanceTimersByTimeAsync(500) })
    expect(result.current.job?.status).toBe('validating')
    await act(async () => { await vi.advanceTimersByTimeAsync(500) })
    expect(result.current.job?.status).toBe('succeeded')
    expect(settled).toHaveBeenCalledWith(expect.objectContaining({ status: 'succeeded' }))
    expect(getMock).toHaveBeenCalledTimes(2)
  })
})
