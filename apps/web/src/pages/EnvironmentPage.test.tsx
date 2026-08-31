import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { KernelLanguage } from '../domain/types'
import { EnvironmentPage } from './EnvironmentPage'

vi.mock('../api/client', () => ({
  api: { environment: vi.fn() },
}))

describe('EnvironmentPage implementation runtimes', () => {
  beforeEach(() => {
    vi.mocked(api.environment).mockImplementation(async (language: KernelLanguage = 'cuda_cpp') => ({
      backend: language,
      healthy: true,
      status: 'healthy',
      gpu_name: 'RTX 4060',
      cuda_runtime_version: '12.4',
      nvcc_version: language === 'cuda_cpp' ? '12.4.1' : undefined,
      python_version: language === 'cuda_cpp' ? undefined : '3.11.10',
      torch_version: language === 'cuda_cpp' ? undefined : '2.5.1',
      torch_cuda_version: language === 'cuda_cpp' ? undefined : '12.4',
      triton_version: language === 'triton_python' ? '3.1.0' : undefined,
    }))
  })

  it('switches to the PyTorch runtime without showing Triton or NVCC as its toolchain', async () => {
    const user = userEvent.setup()
    render(<EnvironmentPage />)

    expect(await screen.findByText('NVCC', { exact: true })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'PyTorch (Python)' }))

    expect(await screen.findByText('Python / PyTorch / Torch CUDA')).toBeInTheDocument()
    expect(screen.getByText('3.11.10 / 2.5.1 / 12.4')).toBeInTheDocument()
    expect(screen.queryByText('NVCC', { exact: true })).not.toBeInTheDocument()
    expect(screen.queryByText('Python / PyTorch / Triton')).not.toBeInTheDocument()
    await waitFor(() => expect(api.environment).toHaveBeenCalledWith('torch_python'))
  })
})
