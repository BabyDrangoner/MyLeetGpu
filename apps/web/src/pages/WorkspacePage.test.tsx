import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { ToastProvider } from '../components/Toast'
import type { Job, ProblemDetail } from '../domain/types'
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

afterEach(cleanup)

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
    torch_python: { language: 'torch_python', display_name: 'PyTorch (Python)', file_extension: '.py', editor_language: 'python', starter_code: '# torch starter', signature: 'def solve(q, k, v)' },
  },
}

function LocationProbe() {
  return <output data-testid="location">{useLocation().search}</output>
}

function renderWorkspace() {
  return render(
    <MemoryRouter initialEntries={['/problems/vector-addition']} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <ToastProvider>
        <Routes><Route path="/problems/:slug" element={<WorkspacePage />} /></Routes>
      </ToastProvider>
    </MemoryRouter>,
  )
}

describe('WorkspacePage language sessions', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    vi.mocked(api.jobs.create).mockReset().mockResolvedValue({ id: 'run-1', language: 'cuda_cpp', action: 'run', status: 'succeeded' })
    vi.mocked(api.jobs.get).mockReset()
    vi.mocked(api.problems.get).mockResolvedValue(problem)
    vi.mocked(api.drafts.get).mockRejectedValue(new Error('尚未保存草稿'))
    vi.mocked(api.drafts.save).mockImplementation(async (problemId, language, source) => ({ problem_id: problemId, language, source, updated_at: new Date().toISOString() }))
    vi.mocked(api.versions.list).mockResolvedValue([])
    vi.mocked(api.versions.findDuplicates).mockResolvedValue([])
  })

  it('flushes and restores independent CUDA, Triton and PyTorch drafts while keeping language in the URL', async () => {
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
    await user.click(screen.getByRole('button', { name: 'PyTorch (Python)' }))

    const torchEditor = await screen.findByLabelText('torch_python editor')
    await waitFor(() => expect(torchEditor).not.toHaveAttribute('readonly'))
    expect(torchEditor).toHaveValue('# torch starter')
    expect(screen.getByTestId('location')).toHaveTextContent('language=torch_python')
    await user.clear(torchEditor)
    await user.type(torchEditor, '# torch edited')
    await user.click(screen.getByRole('button', { name: 'CUDA C++' }))

    await waitFor(() => expect(screen.getByLabelText('cuda_cpp editor')).toHaveValue('// cuda edited'))
    expect(readLocalDraft('vector-addition', 'cuda_cpp')?.source).toBe('// cuda edited')
    expect(readLocalDraft('vector-addition', 'triton_python')?.source).toBe('# triton edited')
    expect(readLocalDraft('vector-addition', 'torch_python')?.source).toBe('# torch edited')
    expect(api.drafts.save).toHaveBeenCalledWith('vector-addition', 'cuda_cpp', '// cuda edited')
    expect(api.drafts.save).toHaveBeenCalledWith('vector-addition', 'triton_python', '# triton edited')
    expect(api.drafts.save).toHaveBeenCalledWith('vector-addition', 'torch_python', '# torch edited')
  })

  it('opens a torch-only problem on PyTorch even when the URL requests CUDA', async () => {
    vi.mocked(api.problems.get).mockResolvedValue({
      ...problem,
      slug: 'multi-head-attention',
      title: '多头注意力',
      default_language: 'torch_python',
      language: 'torch_python',
      starter_code: '# attention starter',
      implementations: {
        torch_python: { language: 'torch_python', display_name: 'PyTorch (Python)', file_extension: '.py', editor_language: 'python', starter_code: '# attention starter', signature: 'def solve(q, k, v)' },
      },
    })

    render(
      <MemoryRouter initialEntries={['/problems/multi-head-attention?language=cuda_cpp']} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <ToastProvider>
          <Routes>
            <Route path="/problems/:slug" element={<><WorkspacePage /><LocationProbe /></>} />
          </Routes>
        </ToastProvider>
      </MemoryRouter>,
    )

    expect(await screen.findByLabelText('torch_python editor')).toHaveValue('# attention starter')
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('language=torch_python'))
    expect(screen.getByRole('button', { name: 'PyTorch (Python)' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'CUDA C++' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '代码检查' })).toBeInTheDocument()
  })

  it('toggles a focused editor layout without losing source or the selected statement view', async () => {
    const user = userEvent.setup()
    renderWorkspace()
    const editor = await screen.findByLabelText('cuda_cpp editor')
    await waitFor(() => expect(editor).not.toHaveAttribute('readonly'))
    await user.clear(editor)
    await user.type(editor, '// focused draft')
    await user.click(screen.getByRole('button', { name: '测量协议' }))
    await user.click(screen.getByRole('button', { name: '专注编码' }))

    expect(screen.queryByRole('region', { name: '题目与测量说明' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '显示题目' })).toHaveAttribute('aria-pressed', 'true')
    expect(editor).toHaveValue('// focused draft')

    await user.click(screen.getByRole('button', { name: '显示题目' }))
    expect(screen.getByRole('region', { name: '题目与测量说明' })).toBeVisible()
    expect(screen.getByRole('button', { name: '测量协议' })).toHaveAttribute('aria-pressed', 'true')
    expect(editor).toHaveValue('// focused draft')
  })

  it('runs the latest source with Ctrl or Command + Enter and expands collapsed output', async () => {
    const user = userEvent.setup()
    renderWorkspace()
    const editor = await screen.findByLabelText('cuda_cpp editor')
    await waitFor(() => expect(editor).not.toHaveAttribute('readonly'))
    await user.click(screen.getByRole('button', { name: '收起任务与输出' }))
    expect(screen.queryByText('等待任务')).not.toBeVisible()

    fireEvent.change(editor, { target: { value: '// latest cuda edit' } })
    fireEvent.keyDown(editor, { key: 'Enter', ctrlKey: true })
    await waitFor(() => expect(api.jobs.create).toHaveBeenCalledWith(expect.objectContaining({ action: 'run', language: 'cuda_cpp', source: '// latest cuda edit' })))
    expect(screen.getByRole('button', { name: '收起任务与输出' })).toHaveAttribute('aria-expanded', 'true')

    await waitFor(() => expect(screen.getByRole('button', { name: '运行样例' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: 'Triton (Python)' }))
    const tritonEditor = await screen.findByLabelText('triton_python editor')
    await waitFor(() => expect(tritonEditor).not.toHaveAttribute('readonly'))
    fireEvent.change(tritonEditor, { target: { value: '# latest triton edit' } })
    fireEvent.keyDown(tritonEditor, { key: 'Enter', metaKey: true })
    await waitFor(() => expect(api.jobs.create).toHaveBeenLastCalledWith(expect.objectContaining({ action: 'run', language: 'triton_python', source: '# latest triton edit' })))
    expect(api.jobs.create).toHaveBeenCalledTimes(2)
  })

  it('ignores run shortcuts while drafts load or a dialog is open', async () => {
    const user = userEvent.setup()
    let releaseDraft!: () => void
    vi.mocked(api.drafts.get).mockImplementationOnce(() => new Promise((_, reject) => { releaseDraft = () => reject(new Error('No remote draft')) }))
    renderWorkspace()
    const editor = await screen.findByLabelText('cuda_cpp editor')
    expect(editor).toHaveAttribute('readonly')
    fireEvent.keyDown(editor, { key: 'Enter', ctrlKey: true })
    expect(api.jobs.create).not.toHaveBeenCalled()

    await act(async () => releaseDraft())
    await waitFor(() => expect(editor).not.toHaveAttribute('readonly'))
    await user.click(screen.getByRole('button', { name: '重置' }))
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Enter', ctrlKey: true })
    expect(api.jobs.create).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: '取消' }))
    await user.click(screen.getByRole('button', { name: '保存为性能版本' }))
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Enter', metaKey: true })
    expect(api.jobs.create).not.toHaveBeenCalled()
  })

  it('prevents repeated submissions and reveals a request failure even if output was collapsed during submission', async () => {
    const user = userEvent.setup()
    let rejectJob!: (error: Error) => void
    vi.mocked(api.jobs.create).mockImplementation(() => new Promise<Job>((_, reject) => { rejectJob = reject }))
    renderWorkspace()
    const editor = await screen.findByLabelText('cuda_cpp editor')
    await waitFor(() => expect(editor).not.toHaveAttribute('readonly'))
    fireEvent.keyDown(editor, { key: 'Enter', ctrlKey: true })
    fireEvent.keyDown(editor, { key: 'Enter', ctrlKey: true })
    fireEvent.keyDown(editor, { key: 'Enter', ctrlKey: true, repeat: true })
    expect(api.jobs.create).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: '运行样例' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: '收起任务与输出' }))
    await act(async () => rejectJob(new Error('GPU worker unavailable')))
    expect(screen.getByRole('button', { name: '收起任务与输出' })).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('region', { name: '任务与输出' })).toHaveTextContent('GPU worker unavailable')
    expect(screen.getByRole('button', { name: '运行样例' })).toBeEnabled()
  })
})
