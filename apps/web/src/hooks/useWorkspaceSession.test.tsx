import type { ReactNode } from 'react'
import { act, cleanup, renderHook } from '@testing-library/react'
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { ProblemDetail } from '../domain/types'
import { readLocalDraft } from '../lib/drafts'
import { useWorkspaceSession } from './useWorkspaceSession'

vi.mock('../api/client', () => ({ api: { drafts: { get: vi.fn(), save: vi.fn() } } }))

const vector: ProblemDetail = {
  slug: 'vector', title: 'Vector', difficulty: 'easy', revision: '1', summary: '', statement_markdown: '',
  default_language: 'cuda_cpp', language: 'cuda_cpp', starter_code: '// CUDA',
  implementations: {
    cuda_cpp: { language: 'cuda_cpp', display_name: 'CUDA', file_extension: '.cu', editor_language: 'cpp', starter_code: '// CUDA' },
    triton_python: { language: 'triton_python', display_name: 'Triton', file_extension: '.py', editor_language: 'python', starter_code: '# Triton' },
  },
}
const attention: ProblemDetail = {
  ...vector, slug: 'attention', title: 'Attention', default_language: 'torch_python', language: 'torch_python',
  implementations: {
    torch_python: { language: 'torch_python', display_name: 'Torch', file_extension: '.py', editor_language: 'python', starter_code: '# Attention' },
  },
}
const wrapper = ({ children }: { children: ReactNode }) => (
  <MemoryRouter initialEntries={['/problems/vector?language=triton_python&panel=code']} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
    {children}
  </MemoryRouter>
)
const useSessionHarness = ({ problemId, problem }: { problemId: string; problem: ProblemDetail | null }) => ({
  ...useWorkspaceSession(problemId, problem),
  location: useLocation(),
  navigate: useNavigate(),
})
const tick = async () => { await act(async () => { await Promise.resolve() }) }

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  vi.mocked(api.drafts.get).mockRejectedValue(new Error('No draft'))
  vi.mocked(api.drafts.save).mockImplementation(async (problem_id, language, source) => ({ problem_id, language, source, updated_at: new Date().toISOString() }))
})
afterEach(cleanup)

describe('workspace route sessions', () => {
  it('waits for the matching problem before loading drafts or normalizing its language', async () => {
    const view = renderHook(useSessionHarness, { initialProps: { problemId: 'vector', problem: vector }, wrapper })
    await tick()
    act(() => view.result.current.draft.updateSource('# saved vector'))
    view.rerender({ problemId: 'attention', problem: vector })
    await tick()
    expect(view.result.current.draft.ready).toBe(false)
    expect(view.result.current.implementation).toBeUndefined()
    expect(api.drafts.get).toHaveBeenCalledTimes(1)
    expect(view.result.current.location.search).toContain('language=triton_python')
    expect(readLocalDraft('vector', 'triton_python')?.source).toBe('# saved vector')

    view.rerender({ problemId: 'attention', problem: attention })
    await tick()
    expect(api.drafts.get).toHaveBeenLastCalledWith('attention', 'torch_python')
    expect(api.drafts.get).not.toHaveBeenCalledWith('attention', 'cuda_cpp')
    expect(api.drafts.get).not.toHaveBeenCalledWith('attention', 'triton_python')
    expect(view.result.current.draft.source).toBe('# Attention')
    expect(view.result.current.location.search).toBe('?language=torch_python&panel=code')
  })

  it('flushes language navigation and preserves independent drafts through browser Back', async () => {
    const view = renderHook(useSessionHarness, { initialProps: { problemId: 'vector', problem: vector }, wrapper })
    await tick()
    act(() => {
      view.result.current.draft.updateSource('# Triton edited')
      view.result.current.selectLanguage('cuda_cpp')
    })
    await tick()
    expect(view.result.current.location.search).toBe('?language=cuda_cpp&panel=code')
    expect(api.drafts.save).toHaveBeenCalledWith('vector', 'triton_python', '# Triton edited')
    act(() => view.result.current.draft.updateSource('// CUDA edited'))
    act(() => view.result.current.navigate(-1))
    await tick()
    expect(view.result.current.language).toBe('triton_python')
    expect(view.result.current.draft.source).toBe('# Triton edited')
    expect(readLocalDraft('vector', 'cuda_cpp')?.source).toBe('// CUDA edited')
  })

  it('rejects unsupported or already selected languages without saving or navigating', async () => {
    const view = renderHook(useSessionHarness, { initialProps: { problemId: 'vector', problem: vector }, wrapper })
    await tick()
    act(() => {
      expect(view.result.current.selectLanguage('torch_python')).toBe(false)
      expect(view.result.current.selectLanguage('triton_python')).toBe(false)
    })
    expect(api.drafts.save).not.toHaveBeenCalled()
    expect(view.result.current.location.search).toBe('?language=triton_python&panel=code')
  })
})
