import { ArrowRight, Search, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { RetryButton, StatusView } from '../components/StatusView'
import { useAsync } from '../hooks/useAsync'
import { difficultyLabel } from '../lib/format'
import { languageMetadata } from '../lib/languages'

type ProblemCategory = 'all' | 'kernel' | 'torch'

const difficultyClass = (difficulty: string) => {
  if (difficulty.includes('困难') || difficulty === 'hard') return 'hard'
  if (difficulty.includes('中等') || difficulty === 'medium') return 'medium'
  return 'easy'
}

export function ProblemListPage() {
  const { data: problems, loading, error, reload } = useAsync(() => api.problems.list(), [])
  const [query, setQuery] = useState('')
  const [difficulty, setDifficulty] = useState('全部')
  const [category, setCategory] = useState<ProblemCategory>('all')

  const categories: { value: ProblemCategory; label: string; count: number }[] = [
    { value: 'all', label: '全部题目', count: problems?.length ?? 0 },
    { value: 'kernel', label: '算子题', count: problems?.filter((problem) => problem.languages?.some((language) => language !== 'torch_python')).length ?? 0 },
    { value: 'torch', label: 'PyTorch 题', count: problems?.filter((problem) => problem.languages?.includes('torch_python')).length ?? 0 },
  ]
  const filtered = useMemo(() => (problems ?? []).filter((problem) => {
    const matchesQuery = `${problem.title} ${problem.slug} ${problem.summary}`.toLowerCase().includes(query.trim().toLowerCase())
    const matchesDifficulty = difficulty === '全部' || difficultyLabel(problem.difficulty) === difficulty
    const matchesCategory = category === 'all' || (category === 'torch'
      ? problem.languages?.includes('torch_python')
      : problem.languages?.some((language) => language !== 'torch_python'))
    return matchesQuery && matchesDifficulty && matchesCategory
  }), [problems, query, difficulty, category])

  const resetFilters = () => { setQuery(''); setDifficulty('全部'); setCategory('all') }

  return (
    <div className="page problems-page">
      <header className="catalog-heading">
        <div>
          <h1>题目</h1>
          <p>从正确实现开始，再探索性能优化。</p>
        </div>
        <span className="catalog-total">{loading ? '正在加载' : `${problems?.length ?? 0} 道练习`}</span>
      </header>

      <section className="list-section" aria-label="题目列表">
        <div className="catalog-toolbar">
          <div className="category-filter" role="group" aria-label="按题型筛选">
            {categories.map((item) => (
              <button key={item.value} type="button" aria-pressed={category === item.value} className={category === item.value ? 'active' : ''} onClick={() => setCategory(item.value)}>
                {item.label}<span>{item.count}</span>
              </button>
            ))}
          </div>
          <div className="list-tools">
            <label className="search-box">
              <Search size={16} aria-hidden="true" />
              <input type="search" aria-label="搜索题目" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索名称或关键词" />
            </label>
            <select aria-label="按难度筛选" value={difficulty} onChange={(event) => setDifficulty(event.target.value)}>
              <option value="全部">全部难度</option>
              <option>入门</option>
              <option>简单</option>
              <option>中等</option>
              <option>困难</option>
            </select>
          </div>
        </div>

        <div className="catalog-description">
          <span role="status">{loading ? '正在读取题目…' : `显示 ${filtered.length} 道题目`}</span>
          {(query || difficulty !== '全部' || category !== 'all') && <button className="text-button" type="button" onClick={resetFilters}><X size={13} />清除筛选</button>}
          <span className="catalog-hint">编写 · 验证 · 比较</span>
        </div>

        {loading && <div className="problem-skeleton-grid" aria-label="正在加载题目">{[1, 2, 3, 4, 5].map((item) => <div className="problem-skeleton" key={item} />)}</div>}
        {!loading && error && <StatusView kind="error" description={error.message} action={<RetryButton onClick={() => void reload()} />} />}
        {!loading && !error && !problems?.length && <StatusView kind="empty" title="还没有题目" description="题目清单为空，请检查 problems 目录与后端启动日志。" />}
        {!loading && !error && !!problems?.length && !filtered.length && <StatusView kind="empty" title="没有匹配的题目" description="试试其他关键词，或清除筛选查看全部题目。" action={<button className="button secondary" type="button" onClick={resetFilters}>清除筛选</button>} />}
        {!loading && !error && filtered.length > 0 && (
          <div className="problem-table">
            <div className="problem-table-heading" aria-hidden="true"><span>#</span><span>题目</span><span>实现语言</span><span>难度</span><span /></div>
            <ul className="problem-list">
              {filtered.map((problem) => (
                <li key={problem.slug}>
                  <Link className="problem-row" to={`/problems/${encodeURIComponent(problem.slug)}`}>
                    <span className="problem-index" aria-hidden="true">{String((problems?.indexOf(problem) ?? 0) + 1).padStart(2, '0')}</span>
                    <div className="problem-copy"><h2>{problem.title}</h2><p>{problem.summary && problem.summary !== problem.title ? problem.summary : problem.slug}</p></div>
                    <div className="problem-languages">{problem.languages?.map((language) => <span key={language}>{languageMetadata[language].shortLabel}</span>)}</div>
                    <span className={`difficulty ${difficultyClass(problem.difficulty)}`}>{difficultyLabel(problem.difficulty)}</span>
                    <ArrowRight className="problem-open" size={16} aria-hidden="true" />
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </div>
  )
}
