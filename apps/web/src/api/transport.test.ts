import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, createApiClient } from './client'
import { request } from './transport'

const fetchMock = vi.fn()

beforeEach(() => {
  vi.useFakeTimers()
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function pendingFetch() {
  fetchMock.mockImplementation((_url: string, options: RequestInit) => new Promise((_resolve, reject) => {
    const rejectAbort = () => reject(new DOMException('Aborted', 'AbortError'))
    if (options.signal?.aborted) rejectAbort()
    else options.signal?.addEventListener('abort', rejectAbort, { once: true })
  }))
}

describe('HTTP transport', () => {
  it('keeps the deadline active when the caller supplies a cancellation signal', async () => {
    pendingFetch()
    const caller = new AbortController()
    const result = request('/slow', { signal: caller.signal }, 50)
    const assertion = expect(result).rejects.toMatchObject({ status: 408, message: expect.stringContaining('超时') })
    await vi.advanceTimersByTimeAsync(50)
    await assertion
    expect(caller.signal.aborted).toBe(false)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('forwards caller cancellation and distinguishes it from timeout', async () => {
    pendingFetch()
    const caller = new AbortController()
    const cleanup = vi.spyOn(caller.signal, 'removeEventListener')
    const result = request('/cancel', { signal: caller.signal })
    const assertion = expect(result).rejects.toMatchObject({ status: 499, message: '请求已取消。' })
    caller.abort()
    await assertion
    expect(cleanup).toHaveBeenCalledWith('abort', expect.any(Function))
    expect(vi.getTimerCount()).toBe(0)
  })

  it('honors an already cancelled caller', async () => {
    pendingFetch()
    const caller = new AbortController()
    caller.abort()
    await expect(request('/cancel', { signal: caller.signal })).rejects.toMatchObject({ status: 499 })
    expect(fetchMock.mock.calls[0][1].signal.aborted).toBe(true)
  })

  it('supports Headers objects and tuple headers without losing JSON defaults', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }))
    await request('/empty', { method: 'POST', body: '{}', headers: new Headers({ 'X-Request-Id': 'abc' }) })
    expect(fetchMock.mock.calls[0][1].headers.get('accept')).toBe('application/json')
    expect(fetchMock.mock.calls[0][1].headers.get('content-type')).toBe('application/json')
    expect(fetchMock.mock.calls[0][1].headers.get('x-request-id')).toBe('abc')
    await request('/empty', { headers: [['Accept', 'text/plain']] })
    expect(fetchMock.mock.calls[1][1].headers.get('accept')).toBe('text/plain')
  })

  it('normalizes text errors and network failures with the public ApiError type', async () => {
    fetchMock.mockResolvedValueOnce(new Response('maintenance', { status: 503 }))
    await expect(request('/offline')).rejects.toEqual(new ApiError('maintenance', 503, 'maintenance'))
    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'))
    await expect(request('/offline')).rejects.toMatchObject({ name: 'ApiError', message: 'Failed to fetch', status: 0 })
    expect(vi.getTimerCount()).toBe(0)
  })

  it('composes resource clients over an injected transport without using fetch', async () => {
    const transport = vi.fn().mockResolvedValue({ items: [{ id: 'a/b', title: 'A' }] })
    const client = createApiClient(transport)
    expect(await client.problems.list()).toEqual([expect.objectContaining({ slug: 'a/b', title: 'A' })])
    expect(transport).toHaveBeenCalledWith('/problems')
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
