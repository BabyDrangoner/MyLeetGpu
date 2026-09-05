import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ThemeProvider, useTheme } from './useTheme'

beforeEach(() => {
  localStorage.clear()
  delete document.documentElement.dataset.theme
  document.documentElement.style.colorScheme = ''
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('theme preference', () => {
  it('defaults to light and keeps standalone components usable without a provider', () => {
    const { result } = renderHook(useTheme)
    expect(result.current.theme).toBe('light')
    expect(() => result.current.toggleTheme()).not.toThrow()
  })

  it('restores dark mode and synchronizes the page before toggling back to light', () => {
    localStorage.setItem('myleetgpu:theme', 'dark')
    const { result, unmount } = renderHook(useTheme, { wrapper: ThemeProvider })

    expect(result.current.theme).toBe('dark')
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(document.documentElement.style.colorScheme).toBe('dark')
    act(() => result.current.toggleTheme())
    expect(result.current.theme).toBe('light')
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(document.documentElement.style.colorScheme).toBe('light')
    expect(localStorage.getItem('myleetgpu:theme')).toBe('light')

    unmount()
    const restored = renderHook(useTheme, { wrapper: ThemeProvider })
    expect(restored.result.current.theme).toBe('light')
  })

  it('ignores an invalid saved preference and persists the user’s next choice', () => {
    localStorage.setItem('myleetgpu:theme', 'unexpected')
    const { result } = renderHook(useTheme, { wrapper: ThemeProvider })
    expect(result.current.theme).toBe('light')
    act(() => result.current.toggleTheme())
    expect(localStorage.getItem('myleetgpu:theme')).toBe('dark')
  })

  it('still switches themes when browser storage cannot be read or written', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('Storage denied') })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('Storage denied') })
    const { result } = renderHook(useTheme, { wrapper: ThemeProvider })
    expect(result.current.theme).toBe('light')
    act(() => result.current.toggleTheme())
    expect(result.current.theme).toBe('dark')
    expect(document.documentElement.dataset.theme).toBe('dark')
  })
})
