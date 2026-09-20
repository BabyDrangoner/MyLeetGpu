import { StrictMode, type ReactNode } from 'react'
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { Draft, KernelLanguage } from '../domain/types'
import { readLocalDraft, saveLocalDraft } from '../lib/drafts'
import { useWorkspaceDraft } from './useWorkspaceDraft'

vi.mock('../api/client', () => ({ api: { drafts: { get: vi.fn(), save: vi.fn() } } }))

const initial = { problemId: 'vector-addition', language: 'cuda_cpp' as KernelLanguage, starterCode: '// cuda starter' as string | undefined }
const remoteDraft = (source: string, updated_at = '2030-01-01T00:00:00Z'): Draft => ({
  problem_id: initial.problemId, language: initial.language, source, updated_at,
})
const tick = async () => { await act(async () => { await Promise.resolve() }) }

beforeEach(() => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  localStorage.clear()
  vi.mocked(api.drafts.get).mockRejectedValue(new Error('No remote draft'))
  vi.mocked(api.drafts.save).mockImplementation(async (problem_id, language, source) => ({ problem_id, language, source, updated_at: '2026-09-06T00:00:00Z' }))
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe('workspace draft sessions', () => {
  it('does not initialize or write a draft until the implementation is available', async () => {
    const view = renderHook(useWorkspaceDraft, { initialProps: { ...initial, starterCode: undefined } })
    expect(view.result.current.ready).toBe(false)
    expect(api.drafts.get).not.toHaveBeenCalled()
    view.unmount()
    expect(readLocalDraft(initial.problemId, initial.language)).toBeNull()
  })

  it('flushes each session to its own key through direct route changes and pagehide', async () => {
    const view = renderHook(useWorkspaceDraft, { initialProps: initial })
    await tick()
    act(() => view.result.current.updateSource('// vector CUDA edit'))
    view.rerender({ ...initial, language: 'triton_python', starterCode: '# triton starter' })
    await tick()
    expect(readLocalDraft(initial.problemId, 'cuda_cpp')?.source).toBe('// vector CUDA edit')
    act(() => view.result.current.updateSource('# vector Triton edit'))
    view.rerender({ ...initial, problemId: 'softmax', starterCode: '// softmax starter' })
    await tick()
    expect(readLocalDraft(initial.problemId, 'triton_python')?.source).toBe('# vector Triton edit')
    act(() => view.result.current.updateSource('// softmax edit'))
    act(() => window.dispatchEvent(new Event('pagehide')))
    expect(readLocalDraft('softmax', 'cuda_cpp')?.source).toBe('// softmax edit')
    expect(readLocalDraft(initial.problemId, 'cuda_cpp')?.source).toBe('// vector CUDA edit')
    view.unmount()
    expect(readLocalDraft(initial.problemId, 'triton_python')?.source).toBe('# vector Triton edit')
  })

  it('ignores a late remote read after navigating to another problem', async () => {
    let resolve!: (draft: Draft) => void
    vi.mocked(api.drafts.get).mockImplementationOnce(() => new Promise((accept) => { resolve = accept }))
    const view = renderHook(useWorkspaceDraft, { initialProps: initial })
    view.rerender({ ...initial, problemId: 'softmax', starterCode: '// softmax starter' })
    await tick()
    act(() => view.result.current.updateSource('// softmax edited'))
    await act(async () => resolve(remoteDraft('// old remote source')))
    expect(view.result.current.source).toBe('// softmax edited')
    expect(view.result.current.ready).toBe(true)
    expect(readLocalDraft(initial.problemId, initial.language)).toBeNull()
  })

  it('restores a newer empty remote draft instead of resurrecting starter code', async () => {
    vi.mocked(api.drafts.get).mockResolvedValue(remoteDraft(''))
    const view = renderHook(useWorkspaceDraft, { initialProps: initial })
    await tick()
    expect(view.result.current.source).toBe('')
    expect(view.result.current.ready).toBe(true)
    expect(readLocalDraft(initial.problemId, initial.language)?.source).toBe('')
  })

  it('keeps a newer local draft and does not replace an edit with a late remote read', async () => {
    saveLocalDraft(initial.problemId, initial.language, '// newer local')
    vi.mocked(api.drafts.get).mockResolvedValueOnce(remoteDraft('// old remote', '2020-01-01T00:00:00Z'))
    const view = renderHook(useWorkspaceDraft, { initialProps: initial })
    await tick()
    expect(view.result.current.source).toBe('// newer local')
    view.unmount()

    let resolve!: (draft: Draft) => void
    vi.mocked(api.drafts.get).mockImplementationOnce(() => new Promise((accept) => { resolve = accept }))
    const pending = renderHook(useWorkspaceDraft, { initialProps: initial })
    act(() => pending.result.current.updateSource('// immediate edit'))
    await act(async () => resolve(remoteDraft('// late remote')))
    expect(pending.result.current.source).toBe('// immediate edit')
  })

  it('debounces local and remote saves and keeps the immediate submission snapshot current', async () => {
    const view = renderHook(useWorkspaceDraft, { initialProps: initial })
    await tick()
    act(() => {
      view.result.current.updateSource('// edit 1')
      view.result.current.updateSource('// edit 2')
      expect(view.result.current.getSource()).toBe('// edit 2')
    })
    await act(async () => vi.advanceTimersByTimeAsync(349))
    expect(readLocalDraft(initial.problemId, initial.language)).toBeNull()
    await act(async () => vi.advanceTimersByTimeAsync(1))
    expect(readLocalDraft(initial.problemId, initial.language)?.source).toBe('// edit 2')
    expect(api.drafts.save).not.toHaveBeenCalled()
    await act(async () => vi.advanceTimersByTimeAsync(750))
    expect(api.drafts.save).toHaveBeenCalledTimes(1)
    expect(api.drafts.save).toHaveBeenCalledWith(initial.problemId, initial.language, '// edit 2')
    expect(view.result.current.status).toBe('saved')
  })

  it('does not show an obsolete save response as confirmation for a newer edit', async () => {
    let resolve!: (draft: Draft) => void
    vi.mocked(api.drafts.save).mockImplementationOnce(() => new Promise((accept) => { resolve = accept }))
    const view = renderHook(useWorkspaceDraft, { initialProps: initial })
    await tick()
    act(() => view.result.current.updateSource('// old edit'))
    await act(async () => vi.advanceTimersByTimeAsync(1_100))
    act(() => view.result.current.updateSource('// new edit'))
    await act(async () => resolve(remoteDraft('// old edit')))
    expect(view.result.current.source).toBe('// new edit')
    expect(view.result.current.status).not.toBe('saved')
    expect(view.result.current.savedAt).not.toBe('2030-01-01T00:00:00Z')
  })

  it('does not leak a late save result into a different language session', async () => {
    let reject!: (error: Error) => void
    vi.mocked(api.drafts.save).mockImplementationOnce(() => new Promise((_, decline) => { reject = decline }))
    const view = renderHook(useWorkspaceDraft, { initialProps: initial })
    await tick()
    await act(async () => vi.advanceTimersByTimeAsync(1_100))
    vi.mocked(api.drafts.get).mockResolvedValueOnce({ ...remoteDraft('# loaded'), language: 'triton_python' })
    view.rerender({ ...initial, language: 'triton_python', starterCode: '# starter' })
    await tick()
    await act(async () => reject(new Error('Old session failed')))
    expect(view.result.current.source).toBe('# loaded')
    expect(view.result.current.status).toBe('saved')
  })

  it('flushes locally before navigation even when the remote save fails', async () => {
    vi.mocked(api.drafts.save).mockRejectedValue(new Error('Offline'))
    const view = renderHook(useWorkspaceDraft, { initialProps: initial })
    await tick()
    act(() => {
      view.result.current.updateSource('// navigate now')
      view.result.current.flush()
    })
    expect(readLocalDraft(initial.problemId, initial.language)?.source).toBe('// navigate now')
    expect(api.drafts.save).toHaveBeenCalledWith(initial.problemId, initial.language, '// navigate now')
    await tick()
  })

  it('preserves the latest draft under StrictMode effect setup and cleanup', async () => {
    const wrapper = ({ children }: { children: ReactNode }) => <StrictMode>{children}</StrictMode>
    const view = renderHook(useWorkspaceDraft, { initialProps: initial, wrapper })
    await tick()
    act(() => view.result.current.updateSource('// strict mode edit'))
    view.unmount()
    expect(readLocalDraft(initial.problemId, initial.language)?.source).toBe('// strict mode edit')
    await act(async () => vi.advanceTimersByTimeAsync(2_000))
    expect(api.drafts.save).not.toHaveBeenCalled()
  })
})
