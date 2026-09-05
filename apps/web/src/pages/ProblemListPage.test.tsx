import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { ProblemListPage } from './ProblemListPage'

vi.mock('../api/client', () => ({
  api: { problems: { list: vi.fn() } },
}))

const listMock = vi.mocked(api.problems.list)

describe('ProblemListPage', () => {
  beforeEach(() => listMock.mockReset())
  afterEach(cleanup)

  it('loads and filters original problems', async () => {
    listMock.mockResolvedValue([
      { slug: 'vector-add', title: '向量加法', difficulty: '入门', revision: '1', summary: '逐元素相加' },
      { slug: 'reduction', title: '并行归约', difficulty: '中等', revision: '1', summary: '求和归约' },
    ])
    const user = userEvent.setup()
    render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><ProblemListPage /></MemoryRouter>)
    expect(await screen.findByText('向量加法')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('显示 2 道题目')
    await user.type(screen.getByLabelText('搜索题目'), '归约')
    expect(screen.queryByText('向量加法')).not.toBeInTheDocument()
    expect(screen.getByText('并行归约')).toBeInTheDocument()
  })

  it('combines category, difficulty and search filters and can reset an empty result', async () => {
    listMock.mockResolvedValue([
      { slug: 'top-k', title: 'Top-K', difficulty: 'hard', revision: '1', summary: '逐行选择', languages: ['cuda_cpp', 'triton_python'] },
      { slug: 'mha', title: '多头注意力', difficulty: 'medium', revision: '1', summary: '实现 attention', languages: ['torch_python'] },
      { slug: 'gqa', title: '分组注意力', difficulty: 'hard', revision: '1', summary: '实现 GQA attention', languages: ['torch_python'] },
    ])
    const user = userEvent.setup()
    render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><ProblemListPage /></MemoryRouter>)
    await screen.findByText('多头注意力')
    await user.click(screen.getByRole('button', { name: /PyTorch 题/ }))
    expect(screen.queryByText('Top-K')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /PyTorch 题/ })).toHaveAttribute('aria-pressed', 'true')
    await user.selectOptions(screen.getByLabelText('按难度筛选'), '困难')
    expect(screen.queryByText('多头注意力')).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: /分组注意力/ })).toHaveAttribute('href', '/problems/gqa')
    await user.type(screen.getByLabelText('搜索题目'), 'top-k')
    expect(screen.getByText('没有匹配的题目')).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: '清除筛选' })[0])
    expect(screen.getAllByRole('link')).toHaveLength(3)
    expect(screen.getByRole('status')).toHaveTextContent('显示 3 道题目')
  })

  it('renders an explicit empty state', async () => {
    listMock.mockResolvedValue([])
    render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><ProblemListPage /></MemoryRouter>)
    expect(await screen.findByText('还没有题目')).toBeInTheDocument()
  })
})
