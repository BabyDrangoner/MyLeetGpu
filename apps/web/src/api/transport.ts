const API_ROOT = '/api'

export class ApiError extends Error {
  readonly status: number
  readonly details: unknown

  constructor(message: string, status: number, details?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.details = details
  }
}

export type HttpTransport = <T>(path: string, init?: RequestInit, timeoutMs?: number) => Promise<T>

function errorMessage(payload: unknown, status: number) {
  const record = payload !== null && typeof payload === 'object' ? payload as Record<string, unknown> : {}
  const nested = record.error !== null && typeof record.error === 'object' ? record.error as Record<string, unknown> : {}
  const detail = record.detail ?? nested.message ?? payload
  return new ApiError(typeof detail === 'string' ? detail : `请求失败（HTTP ${status}）`, status, detail)
}

// Transport owns HTTP concerns only; wire-format adaptation belongs to mappers.
export const request: HttpTransport = async <T>(path: string, init?: RequestInit, timeoutMs = 15_000): Promise<T> => {
  const controller = new AbortController()
  const abortFromCaller = () => controller.abort(init?.signal?.reason)
  if (init?.signal?.aborted) abortFromCaller()
  else init?.signal?.addEventListener('abort', abortFromCaller, { once: true })
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    const headers = new Headers({ Accept: 'application/json' })
    if (init?.body) headers.set('Content-Type', 'application/json')
    new Headers(init?.headers).forEach((value, key) => headers.set(key, value))
    const response = await fetch(`${API_ROOT}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
    })
    const contentType = response.headers.get('content-type') ?? ''
    const payload: unknown = response.status === 204
      ? undefined
      : contentType.includes('application/json')
        ? await response.json()
        : await response.text()
    if (!response.ok) throw errorMessage(payload, response.status)
    return payload as T
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (init?.signal?.aborted) throw new ApiError('请求已取消。', 499)
    if (controller.signal.aborted) throw new ApiError('连接 API 超时，请确认服务仍在运行。', 408)
    throw new ApiError(error instanceof Error ? error.message : '无法连接本地 API。', 0)
  } finally {
    window.clearTimeout(timer)
    init?.signal?.removeEventListener('abort', abortFromCaller)
  }
}
