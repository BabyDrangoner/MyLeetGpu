import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { JobPanel } from './JobPanel'

describe('JobPanel', () => {
  it('shows timeout as a distinct terminal state', () => {
    render(<JobPanel job={{ id: 'job-timeout', action: 'run', status: 'timed_out', error: { message: '运行超过 5 秒' } }} />)
    expect(screen.getByText('已超时')).toBeInTheDocument()
    expect(screen.getByText('运行超过 5 秒')).toBeInTheDocument()
  })

  it('renders cleaned compiler diagnostics and per-case status', () => {
    render(<JobPanel job={{
      id: 'job-failed',
      action: 'run',
      status: 'failed',
      diagnostics: 'solution.cu:12: error: expected ;',
      result: { cases: [{ name: '样例 1', passed: false, error_type: 'wrong_answer', message: '结果不匹配' }] },
    }} />)
    expect(screen.getByText('样例 1')).toBeInTheDocument()
    expect(screen.getByText(/wrong_answer/)).toBeInTheDocument()
    expect(screen.getByText('solution.cu:12: error: expected ;')).toBeInTheDocument()
  })
})
