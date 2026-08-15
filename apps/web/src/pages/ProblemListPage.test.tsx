import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { ProblemListPage } from './ProblemListPage'

vi.mock('../api/client', () => ({
  api: { problems: { list: vi.fn() } },
}))

const listMock = vi.mocked(api.problems.list)

describe('ProblemListPage', () => {
  beforeEach(() => listMock.mockReset())

  it('loads and filters original problems', async () => {
    listMock.mockResolvedValue([
      { slug: 'vector-add', title: '向量加法', difficulty: '入门', revision: '1', summary: '逐元素相加' },
      { slug: 'reduction', title: '并行归约', difficulty: '中等', revision: '1', summary: '求和归约' },
    ])
    const user = userEvent.setup()
    render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><ProblemListPage /></MemoryRouter>)
    expect(await screen.findByText('向量加法')).toBeInTheDocument()
    await user.type(screen.getByLabelText('搜索题目'), '归约')
    expect(screen.queryByText('向量加法')).not.toBeInTheDocument()
    expect(screen.getByText('并行归约')).toBeInTheDocument()
  })

  it('renders an explicit empty state', async () => {
    listMock.mockResolvedValue([])
    render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><ProblemListPage /></MemoryRouter>)
    expect(await screen.findByText('还没有题目')).toBeInTheDocument()
  })
})
