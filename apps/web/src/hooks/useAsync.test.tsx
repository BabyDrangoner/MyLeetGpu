import { renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { useAsync } from './useAsync'

describe('useAsync', () => {
  it('turns a rejected request into displayable state', async () => {
    const loader = vi.fn(async () => {
      throw new Error('API 未启动')
    })
    const { result } = renderHook(() => useAsync(loader, []))
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.error?.message).toBe('API 未启动')
  })
})
