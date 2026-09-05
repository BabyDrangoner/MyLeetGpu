import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { JobPanel } from './JobPanel'

afterEach(cleanup)

describe('JobPanel', () => {
  it('shows a new request error alongside the previous task result', () => {
    render(<JobPanel job={{ id: 'last-run', language: 'cuda_cpp', action: 'run', status: 'succeeded', result: { summary: '上次样例全部通过' } }} requestError={new Error('无法连接执行服务')} />)
    expect(screen.getByRole('alert')).toHaveTextContent('无法连接执行服务')
    expect(screen.getByText('上次样例全部通过')).toBeInTheDocument()
  })

  it('shows timeout as a distinct terminal state', () => {
    render(<JobPanel job={{ id: 'job-timeout', language: 'cuda_cpp', action: 'run', status: 'timed_out', error: { message: '运行超过 5 秒' } }} />)
    expect(screen.getByText('已超时')).toBeInTheDocument()
    expect(screen.getByText('运行超过 5 秒')).toBeInTheDocument()
  })

  it('renders cleaned compiler diagnostics and per-case status', () => {
    render(<JobPanel job={{
      id: 'job-failed',
      language: 'cuda_cpp',
      action: 'run',
      status: 'failed',
      diagnostics: 'solution.cu:12: error: expected ;',
      result: { cases: [{ name: '样例 1', passed: false, error_type: 'wrong_answer', message: '结果不匹配' }] },
    }} />)
    expect(screen.getByText('样例 1')).toBeInTheDocument()
    expect(screen.getByText(/wrong_answer/)).toBeInTheDocument()
    expect(screen.getByText('solution.cu:12: error: expected ;')).toBeInTheDocument()
  })

  it('labels Triton diagnostics without calling them NVCC output', () => {
    render(<JobPanel job={{ id: 'job-triton', language: 'triton_python', action: 'compile', status: 'failed', diagnostics: 'solution.py:4: SyntaxError' }} />)
    expect(screen.getByText('Triton / Python 诊断')).toBeInTheDocument()
    expect(screen.queryByText('NVCC 诊断')).not.toBeInTheDocument()
  })

  it('labels PyTorch checks without falling back to CUDA or Triton', () => {
    render(<JobPanel job={{ id: 'job-torch', language: 'torch_python', action: 'compile', status: 'failed', diagnostics: 'solution.py:8: NameError' }} />)
    expect(screen.getByText('Python / PyTorch 诊断')).toBeInTheDocument()
    expect(screen.getByText('PyTorch')).toBeInTheDocument()
    expect(screen.queryByText('NVCC 诊断')).not.toBeInTheDocument()
    expect(screen.queryByText('Triton / Python 诊断')).not.toBeInTheDocument()
  })
})
