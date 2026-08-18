import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { ToastProvider } from '../components/Toast'
import type { ProblemDetail } from '../domain/types'
import { readLocalDraft } from '../lib/drafts'
import { WorkspacePage } from './WorkspacePage'

vi.mock('../api/client', () => ({
  api: {
    problems: { get: vi.fn() },
    drafts: { get: vi.fn(), save: vi.fn() },
    versions: { list: vi.fn(), findDuplicates: vi.fn() },
    jobs: { create: vi.fn(), get: vi.fn() },
  },
}))

vi.mock('../components/CodeEditor', () => ({
  CodeEditor: ({ value, language, readOnly, onChange }: { value: string; language: string; readOnly?: boolean; onChange?: (value: string) => void }) => (
    <textarea aria-label={`${language} editor`} value={value} readOnly={readOnly} onChange={(event) => onChange?.(event.target.value)} />
  ),
}))

const problem: ProblemDetail = {
  slug: 'vector-addition',
  title: '向量加法',
  difficulty: 'easy',
  revision: '2',
  summary: '双语言题目',
  statement_markdown: '# 题目',
  default_language: 'cuda_cpp',
  language: 'cuda_cpp',
  starter_code: '// cuda starter',
  implementations: {
    cuda_cpp: { language: 'cuda_cpp', display_name: 'CUDA C++', file_extension: '.cu', editor_language: 'cpp', starter_code: '// cuda starter', signature: 'void solve()' },
    triton_python: { language: 'triton_python', display_name: 'Triton Python', file_extension: '.py', editor_language: 'python', starter_code: '# triton starter', signature: 'def solve()' },
  },
}

function LocationProbe() {
  return <output data-testid="location">{useLocation().search}</output>
}

describe('WorkspacePage language sessions', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.mocked(api.problems.get).mockResolvedValue(problem)
    vi.mocked(api.drafts.get).mockRejectedValue(new Error('尚未保存草稿'))
    vi.mocked(api.drafts.save).mockImplementation(async (problemId, language, source) => ({ problem_id: problemId, language, source, updated_at: new Date().toISOString() }))
    vi.mocked(api.versions.list).mockResolvedValue([])
    vi.mocked(api.versions.findDuplicates).mockResolvedValue([])
  })

  it('flushes and restores independent CUDA and Triton drafts while keeping language in the URL', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={['/problems/vector-addition']} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <ToastProvider>
          <Routes>
            <Route path="/problems/:slug" element={<><WorkspacePage /><LocationProbe /></>} />
          </Routes>
        </ToastProvider>
      </MemoryRouter>,
    )

    const cudaEditor = await screen.findByLabelText('cuda_cpp editor')
    await waitFor(() => expect(cudaEditor).not.toHaveAttribute('readonly'))
    await user.clear(cudaEditor)
    await user.type(cudaEditor, '// cuda edited')
    await user.click(screen.getByRole('button', { name: 'Triton (Python)' }))

    const tritonEditor = await screen.findByLabelText('triton_python editor')
    await waitFor(() => expect(tritonEditor).not.toHaveAttribute('readonly'))
    expect(tritonEditor).toHaveValue('# triton starter')
    expect(screen.getByTestId('location')).toHaveTextContent('language=triton_python')
    await user.clear(tritonEditor)
    await user.type(tritonEditor, '# triton edited')
    await user.click(screen.getByRole('button', { name: 'CUDA C++' }))

    await waitFor(() => expect(screen.getByLabelText('cuda_cpp editor')).toHaveValue('// cuda edited'))
    expect(readLocalDraft('vector-addition', 'cuda_cpp')?.source).toBe('// cuda edited')
    expect(readLocalDraft('vector-addition', 'triton_python')?.source).toBe('# triton edited')
    expect(api.drafts.save).toHaveBeenCalledWith('vector-addition', 'cuda_cpp', '// cuda edited')
    expect(api.drafts.save).toHaveBeenCalledWith('vector-addition', 'triton_python', '# triton edited')
  })
})
