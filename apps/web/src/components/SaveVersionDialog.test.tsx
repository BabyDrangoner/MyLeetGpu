import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { SavedVersion } from '../domain/types'
import { sourceHash } from '../lib/hash'
import { SaveVersionDialog } from './SaveVersionDialog'

vi.mock('../api/client', () => ({
  api: { versions: { findDuplicates: vi.fn().mockResolvedValue([]) } },
}))
vi.mock('../lib/hash', () => ({ sourceHash: vi.fn().mockResolvedValue('same-hash') }))

const existing: SavedVersion = {
  id: 'v1',
  problem_id: 'vector-add',
  problem_revision: '1',
  language: 'cuda_cpp',
  name: '初始实现',
  source_hash: 'same-hash',
  source_code: '// snapshot',
  created_at: '2026-01-01T00:00:00Z',
  correctness_status: 'passed',
  benchmark_runs: [],
}

describe('SaveVersionDialog', () => {
  beforeEach(() => {
    vi.mocked(sourceHash).mockResolvedValue('same-hash')
    vi.mocked(api.versions.findDuplicates).mockResolvedValue([])
  })

  it('warns about duplicates, requires acknowledgement, and submits the immutable snapshot', async () => {
    const onSave = vi.fn()
    const user = userEvent.setup()
    render(
      <SaveVersionDialog
        open
        problemId="vector-add"
        language="cuda_cpp"
        snapshot="// snapshot"
        currentSource="// edited after click"
        existingVersions={[existing]}
        busy={false}
        onClose={vi.fn()}
        onSave={onSave}
      />,
    )

    expect(screen.getByText('编辑器已有新修改；本次仍保存上面锁定的快照。')).toBeInTheDocument()
    expect(await screen.findByText('检测到相同源码')).toBeInTheDocument()
    await user.type(screen.getByPlaceholderText('例如：共享内存分块 v2'), '复测版本')
    const submit = screen.getByRole('button', { name: '验证、测速并保存' })
    expect(submit).toBeDisabled()
    await user.click(screen.getByLabelText('我了解，仍要保存重复源码'))
    expect(submit).toBeEnabled()
    await user.click(submit)
    await waitFor(() => expect(onSave).toHaveBeenCalledWith(expect.objectContaining({
      name: '复测版本',
      source: '// snapshot',
      language: 'cuda_cpp',
      allowDuplicate: true,
    })))
  })
})
