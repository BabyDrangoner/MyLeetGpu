import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { EnvironmentSnapshot, ExecutionSettings as Settings, KernelLanguage } from '../domain/types'
import { ExecutionSettingsProvider, useExecutionSettings } from '../hooks/useExecutionSettings'
import { ExecutionSettings } from './ExecutionSettings'

vi.mock('../api/client', () => ({ api: { executionSettings: { get: vi.fn(), save: vi.fn(), probe: vi.fn() } } }))

const defaults: Settings = { target: 'local', colab_acknowledged: false, colab: { ssh_host: 'colab-vscode', remote_root: '/content/project/myleetgpu-runner', isolation: 'trusted-native' }, updated_at: null }

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((res) => { resolve = res })
  return { promise, resolve }
}

function Harness() {
  const [language, setLanguage] = useState<KernelLanguage>('cuda_cpp')
  const execution = useExecutionSettings()
  return <><button onClick={() => setLanguage('triton_python')}>切换测试语言</button><button onClick={() => void execution?.reload()}>重新读取</button><ExecutionSettings language={language} /></>
}

function mount() {
  return render(<MemoryRouter><ExecutionSettingsProvider><Harness /></ExecutionSettingsProvider></MemoryRouter>)
}

async function ready() {
  await waitFor(() => expect(screen.getByRole('radio', { name: /本地 GPU/ })).toBeEnabled())
}

afterEach(cleanup)
beforeEach(() => {
  vi.resetAllMocks()
  vi.mocked(api.executionSettings.get).mockResolvedValue(defaults)
  vi.mocked(api.executionSettings.save).mockImplementation(async (input) => ({ ...defaults, ...input, updated_at: '2026-09-05T01:00:00Z' }))
  vi.mocked(api.executionSettings.probe).mockImplementation(async (target, backend) => ({ execution_target: target, backend, healthy: true, gpu_name: 'Tesla T4' }))
})

describe('ExecutionSettings', () => {
  it('requires informed consent and explicit save before changing the active execution target', async () => {
    const user = userEvent.setup()
    mount()
    await ready()
    await user.click(screen.getByRole('radio', { name: /Google Colab/ }))
    expect(screen.getByTestId('active-execution-target')).toHaveTextContent('当前：本地 GPU')
    expect(screen.getByRole('button', { name: '保存设置' })).toBeDisabled()
    expect(screen.getByText('colab-vscode')).toBeInTheDocument()
    expect(screen.getByText('/content/project/myleetgpu-runner')).toBeInTheDocument()
    await user.click(screen.getByRole('checkbox', { name: /Colab 不使用 Docker/ }))
    expect(api.executionSettings.save).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: '保存设置' }))
    await waitFor(() => expect(screen.getByTestId('active-execution-target')).toHaveTextContent('当前：Colab'))
    expect(api.executionSettings.save).toHaveBeenCalledWith({ target: 'colab', colab_acknowledged: true })
    expect(screen.getByText(/已有任务和草稿不受影响/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '保存设置' })).toBeDisabled()
  })

  it('probes the candidate and language without saving or replacing the active target', async () => {
    const user = userEvent.setup()
    mount()
    await ready()
    await user.click(screen.getByRole('radio', { name: /Google Colab/ }))
    await user.click(screen.getByRole('button', { name: '切换测试语言' }))
    await user.click(screen.getByRole('button', { name: '测试连接' }))
    expect(await screen.findByText('Colab · Triton (Python)：连接正常')).toBeInTheDocument()
    expect(api.executionSettings.probe).toHaveBeenCalledWith('colab', 'triton_python')
    expect(api.executionSettings.save).not.toHaveBeenCalled()
    expect(screen.getByTestId('active-execution-target')).toHaveTextContent('本地 GPU')
  })

  it.each(['target', 'language'])('ignores an old probe when the %s changes', async (change) => {
    const pending = deferred<EnvironmentSnapshot>()
    vi.mocked(api.executionSettings.probe).mockReturnValueOnce(pending.promise)
    const user = userEvent.setup()
    mount()
    await ready()
    await user.click(screen.getByRole('radio', { name: /Google Colab/ }))
    await user.click(screen.getByRole('button', { name: '测试连接' }))
    expect(screen.getByRole('button', { name: '正在测试连接' })).toBeDisabled()
    expect(screen.getByText(/最多约 3 分钟/)).toBeInTheDocument()
    if (change === 'target') await user.click(screen.getByRole('radio', { name: /本地 GPU/ }))
    else await user.click(screen.getByRole('button', { name: '切换测试语言' }))
    await act(async () => pending.resolve({ healthy: true, gpu_name: 'STALE RESULT' }))
    expect(screen.queryByText('STALE RESULT')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '测试连接' })).toBeEnabled()
  })

  it('keeps the form usable after a denied save and an offline probe', async () => {
    vi.mocked(api.executionSettings.save).mockRejectedValueOnce(new Error('无权修改设置'))
    vi.mocked(api.executionSettings.probe).mockRejectedValueOnce(new Error('Worker 未在线'))
    const user = userEvent.setup()
    mount()
    await ready()
    await user.click(screen.getByRole('radio', { name: /Google Colab/ }))
    await user.click(screen.getByRole('checkbox'))
    await user.click(screen.getByRole('button', { name: '保存设置' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('无权修改设置')
    expect(screen.getByTestId('active-execution-target')).toHaveTextContent('本地 GPU')
    expect(screen.getByRole('button', { name: '保存设置' })).toBeEnabled()
    await user.click(screen.getByRole('radio', { name: /本地 GPU/ }))
    await user.click(screen.getByRole('button', { name: '测试连接' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Worker 未在线')
    expect(screen.getByRole('button', { name: '测试连接' })).toBeEnabled()
  })

  it('supports retry after settings cannot be read, without assuming a local target', async () => {
    vi.mocked(api.executionSettings.get).mockRejectedValueOnce(new Error('请先登录'))
    const user = userEvent.setup()
    mount()
    expect(await screen.findByRole('alert')).toHaveTextContent('请先登录')
    expect(screen.getByTestId('active-execution-target')).toHaveTextContent('目标未加载')
    expect(screen.getByRole('radio', { name: /Google Colab/ })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: '重试读取设置' }))
    await ready()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('refreshes server settings but preserves an unsaved candidate', async () => {
    const user = userEvent.setup()
    mount()
    await ready()
    await user.click(screen.getByRole('radio', { name: /Google Colab/ }))
    vi.mocked(api.executionSettings.get).mockResolvedValueOnce({ ...defaults, updated_at: '2026-09-06T01:00:00Z' })
    await user.click(screen.getByRole('button', { name: '重新读取' }))
    await waitFor(() => expect(screen.getByTestId('active-execution-target')).not.toHaveTextContent('更新中'))
    expect(screen.getByRole('radio', { name: /Google Colab/ })).toBeChecked()
    expect(screen.getByRole('checkbox')).not.toBeChecked()
    expect(screen.getByTestId('active-execution-target')).toHaveTextContent('本地 GPU')
  })

  it('does not let a stale read overwrite a successful save', async () => {
    const pending = deferred<Settings>()
    const user = userEvent.setup()
    mount()
    await ready()
    vi.mocked(api.executionSettings.get).mockReturnValueOnce(pending.promise)
    await user.click(screen.getByRole('button', { name: '重新读取' }))
    await user.click(screen.getByRole('radio', { name: /Google Colab/ }))
    await user.click(screen.getByRole('checkbox'))
    await user.click(screen.getByRole('button', { name: '保存设置' }))
    await waitFor(() => expect(screen.getByTestId('active-execution-target')).toHaveTextContent('当前：Colab'))
    await act(async () => pending.resolve(defaults))
    expect(screen.getByTestId('active-execution-target')).toHaveTextContent('当前：Colab')
  })
})
