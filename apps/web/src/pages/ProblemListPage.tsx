import { ArrowRight, BookOpen, Gauge, Search, Sparkles } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { RetryButton, StatusView } from '../components/StatusView'
import { useAsync } from '../hooks/useAsync'
import { difficultyLabel } from '../lib/format'

const difficultyClass = (difficulty: string) => {
  if (difficulty.includes('困难') || difficulty === 'hard') return 'hard'
  if (difficulty.includes('中等') || difficulty === 'medium') return 'medium'
  return 'easy'
}

export function ProblemListPage() {
  const { data: problems, loading, error, reload } = useAsync(() => api.problems.list(), [])
  const [query, setQuery] = useState('')
  const [difficulty, setDifficulty] = useState('全部')

  const filtered = useMemo(() => (problems ?? []).filter((problem) => {
    const matchesQuery = `${problem.title} ${problem.slug} ${problem.summary}`.toLowerCase().includes(query.trim().toLowerCase())
    const matchesDifficulty = difficulty === '全部' || difficultyLabel(problem.difficulty) === difficulty
    return matchesQuery && matchesDifficulty
  }), [problems, query, difficulty])

  return (
    <div className="page problems-page">
      <section className="hero-section">
        <div>
          <div className="eyebrow"><Sparkles size={14} /> CUDA KERNEL LAB</div>
          <h1>把想法变成更快的 Kernel</h1>
          <p>在同一套本地环境里编写、验证并严谨比较 CUDA C++ 实现。</p>
        </div>
        <div className="hero-stat-grid">
          <div className="hero-stat"><strong>{problems?.length ?? '—'}</strong><span>原创题目</span></div>
          <div className="hero-stat"><strong>CUDA</strong><span>唯一语言</span></div>
          <div className="hero-stat"><strong>Median</strong><span>核心指标</span></div>
        </div>
      </section>

      <section className="list-section">
        <div className="section-heading-row">
          <div>
            <h2>选择一道题目</h2>
            <p>每道题都提供平台控制的验证与 benchmark harness。</p>
          </div>
          <div className="list-tools">
            <label className="search-box">
              <Search size={16} />
              <input aria-label="搜索题目" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索题目…" />
            </label>
            <select aria-label="按难度筛选" value={difficulty} onChange={(event) => setDifficulty(event.target.value)}>
              <option>全部</option>
              <option>入门</option>
              <option>简单</option>
              <option>中等</option>
              <option>困难</option>
            </select>
          </div>
        </div>

        {loading && <div className="problem-skeleton-grid" aria-label="正在加载题目">{[1, 2, 3].map((item) => <div className="problem-skeleton" key={item} />)}</div>}
        {!loading && error && <StatusView kind="error" description={error.message} action={<RetryButton onClick={() => void reload()} />} />}
        {!loading && !error && !problems?.length && <StatusView kind="empty" title="还没有题目" description="题目清单为空，请检查 problems 目录与后端启动日志。" />}
        {!loading && !error && !!problems?.length && !filtered.length && <StatusView kind="empty" title="没有匹配的题目" description="试试缩短关键词或选择其他难度。" />}
        {!loading && !error && filtered.length > 0 && (
          <div className="problem-grid">
            {filtered.map((problem, index) => (
              <Link className="problem-card" to={`/problems/${encodeURIComponent(problem.slug)}`} key={problem.slug}>
                <div className="problem-card-top">
                  <span className="problem-index">{String(index + 1).padStart(2, '0')}</span>
                  <span className={`difficulty ${difficultyClass(problem.difficulty)}`}>{difficultyLabel(problem.difficulty)}</span>
                </div>
                <div className="problem-glyph" aria-hidden="true">
                  {index % 3 === 0 ? <span className="vector-glyph">A + B → C</span> : index % 3 === 1 ? <span className="matrix-glyph"><i /><i /><i /><i /></span> : <Gauge size={33} />}
                </div>
                <h3>{problem.title}</h3>
                <p>{problem.summary || '编写高效、正确且可验证的 CUDA Kernel。'}</p>
                <div className="problem-meta">
                  <span><BookOpen size={14} /> 修订 {problem.revision}</span>
                  <span className="start-link">开始编码 <ArrowRight size={15} /></span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
