import { AlertTriangle, ArrowLeft, BarChart3, Check, CheckCircle2, Code2, Edit3, GitCompareArrows, LoaderCircle, Play, RefreshCw, Trash2, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api/client'
import { CodeDiff } from '../components/CodeEditor'
import { JobPanel } from '../components/JobPanel'
import { Modal } from '../components/Modal'
import { RetryButton, StatusView } from '../components/StatusView'
import { useToast } from '../components/Toast'
import type { ComparisonResult, Job, SavedVersion } from '../domain/types'
import { useAsync } from '../hooks/useAsync'
import { useJob } from '../hooks/useJob'
import { comparisonMetric, latestBenchmarkRun, localComparability } from '../lib/benchmark'
import { formatDate, formatMetric, formatPercent } from '../lib/format'

const MAX_SELECTED_VERSIONS = 8

function latestRun(version: SavedVersion) {
  return latestBenchmarkRun(version)
}

function shortHash(hash?: string) {
  return hash ? hash.slice(0, 10) : '—'
}

export function VersionsPage() {
  const { slug = '' } = useParams()
  const toast = useToast()
  const problem = useAsync(() => api.problems.get(slug), [slug])
  const versionsState = useAsync(() => api.versions.list(slug), [slug])
  const [selected, setSelected] = useState<string[]>([])
  const [baselineId, setBaselineId] = useState('')
  const [comparison, setComparison] = useState<ComparisonResult | null>(null)
  const [comparisonLoading, setComparisonLoading] = useState(false)
  const [comparisonError, setComparisonError] = useState<Error | null>(null)
  const [editing, setEditing] = useState<SavedVersion | null>(null)
  const [editName, setEditName] = useState('')
  const [editNotes, setEditNotes] = useState('')
  const [savingEdit, setSavingEdit] = useState(false)
  const [deleting, setDeleting] = useState<SavedVersion | null>(null)
  const [deleteBusy, setDeleteBusy] = useState(false)
  const [retestOpen, setRetestOpen] = useState(false)
  const [diffOriginal, setDiffOriginal] = useState('')
  const [diffModified, setDiffModified] = useState('')
  const compareRequest = useRef(0)

  const versions = versionsState.data ?? []
  const selectedVersions = useMemo(
    () => selected.map((id) => versions.find((version) => version.id === id)).filter((version): version is SavedVersion => !!version),
    [selected, versions],
  )
  const localCheck = useMemo(() => localComparability(selectedVersions), [selectedVersions])

  const reloadAfterJob = useCallback((settled: Job) => {
    if (settled.status === 'succeeded') {
      toast.show('统一环境重测已完成，结果已追加到所选版本', 'success')
      setComparison(null)
      void versionsState.reload()
    } else {
      toast.show(settled.status === 'timed_out' ? '重测任务超时' : '重测任务未完成', 'error')
    }
  }, [toast, versionsState.reload])
  const retestJob = useJob(reloadAfterJob)

  useEffect(() => {
    if (!versions.length || selected.length) return
    const defaults = versions.slice(0, 2).map(({ id }) => id)
    setSelected(defaults)
    setBaselineId(defaults[0] ?? '')
    setDiffOriginal(defaults[0] ?? '')
    setDiffModified(defaults[1] ?? defaults[0] ?? '')
  }, [versions, selected.length])

  useEffect(() => {
    if (selected.length < 2 || !baselineId || !selected.includes(baselineId)) {
      setComparison(null)
      setComparisonError(null)
      return
    }
    const current = ++compareRequest.current
    setComparisonLoading(true)
    setComparisonError(null)
    const timer = window.setTimeout(() => {
      void api.versions.compare(slug, selected, baselineId).then((result) => {
        if (current === compareRequest.current) setComparison(result)
      }).catch((error) => {
        if (current === compareRequest.current) {
          setComparison(null)
          setComparisonError(error instanceof Error ? error : new Error('比较失败'))
        }
      }).finally(() => {
        if (current === compareRequest.current) setComparisonLoading(false)
      })
    }, 250)
    return () => window.clearTimeout(timer)
  }, [baselineId, selected, slug, versionsState.data])

  const applySelection = (next: string[]) => {
    setSelected(next)
    if (!next.includes(baselineId)) setBaselineId(next[0] ?? '')
    if (!next.includes(diffOriginal)) setDiffOriginal(next[0] ?? '')
    if (!next.includes(diffModified)) setDiffModified(next[1] ?? next[0] ?? '')
  }

  const toggleSelected = (id: string) => {
    if (!selected.includes(id) && selected.length >= MAX_SELECTED_VERSIONS) {
      toast.show(`一次最多比较 ${MAX_SELECTED_VERSIONS} 个版本`, 'error')
      return
    }
    applySelection(
      selected.includes(id) ? selected.filter((item) => item !== id) : [...selected, id],
    )
  }

  const openEdit = (version: SavedVersion) => {
    setEditing(version)
    setEditName(version.name)
    setEditNotes(version.notes ?? '')
  }

  const saveEdit = async () => {
    if (!editing || !editName.trim()) return
    setSavingEdit(true)
    try {
      await api.versions.update(editing.id, { name: editName.trim(), notes: editNotes.trim() })
      toast.show('版本信息已更新', 'success')
      setEditing(null)
      await versionsState.reload()
    } catch (error) {
      toast.show(error instanceof Error ? error.message : '更新失败', 'error')
    } finally {
      setSavingEdit(false)
    }
  }

  const confirmDelete = async () => {
    if (!deleting) return
    setDeleteBusy(true)
    try {
      await api.versions.remove(deleting.id)
      applySelection(selected.filter((id) => id !== deleting.id))
      toast.show(`已删除「${deleting.name}」`, 'success')
      setDeleting(null)
      await versionsState.reload()
    } catch (error) {
      toast.show(error instanceof Error ? error.message : '删除失败', 'error')
    } finally {
      setDeleteBusy(false)
    }
  }

  const startRetest = async () => {
    setRetestOpen(false)
    try {
      await retestJob.start({ problem_id: slug, action: 'rebenchmark', version_ids: selected })
    } catch (error) {
      toast.show(error instanceof Error ? error.message : '提交重测失败', 'error')
    }
  }

  if (problem.loading || versionsState.loading) return <div className="page"><StatusView kind="loading" title="正在读取性能版本" /></div>
  if (problem.error || versionsState.error || !problem.data) {
    return <div className="page"><StatusView kind="error" description={problem.error?.message ?? versionsState.error?.message} action={<RetryButton onClick={() => { void problem.reload(); void versionsState.reload() }} />} /></div>
  }

  const comparable = comparison?.comparable ?? localCheck.comparable
  const reasons = comparison?.reasons?.length ? comparison.reasons : localCheck.reasons
  const original = versions.find(({ id }) => id === diffOriginal)
  const modified = versions.find(({ id }) => id === diffModified)

  return (
    <div className="page versions-page">
      <div className="page-heading versions-heading">
        <div>
          <Link className="back-link" to={`/problems/${encodeURIComponent(slug)}`}><ArrowLeft size={15} />返回编辑器</Link>
          <h1>性能版本 · {problem.data.title}</h1>
          <p>选择至少两个版本，以同一个 baseline 查看逐规模 speedup。</p>
        </div>
        <button className="button accent" type="button" disabled={selected.length < 1 || retestJob.busy} onClick={() => setRetestOpen(true)}>
          {retestJob.busy ? <LoaderCircle className="spin" size={16} /> : <RefreshCw size={16} />}
          统一环境重测所选版本
        </button>
      </div>

      {!versions.length ? (
        <StatusView
          kind="empty"
          title="还没有性能版本"
          description="回到编辑器，点击“保存为性能版本”。平台会在验证与 benchmark 都成功后创建第一条记录。"
          action={<Link className="button primary" to={`/problems/${encodeURIComponent(slug)}`}>去保存第一个版本</Link>}
        />
      ) : (
        <>
          <section className="version-picker panel">
            <header className="panel-heading">
              <div><h2>已保存版本</h2><p>已选择 {selected.length} / {Math.min(versions.length, MAX_SELECTED_VERSIONS)}；圆点表示 baseline。</p></div>
              <span className="immutable-label">代码与测量快照不可变</span>
            </header>
            <div className="version-list">
              {versions.map((version) => {
                const checked = selected.includes(version.id)
                const run = latestRun(version)
                return (
                  <article className={`version-row ${checked ? 'selected' : ''}`} key={version.id}>
                    <label className="version-check">
                      <input type="checkbox" checked={checked} disabled={!checked && selected.length >= MAX_SELECTED_VERSIONS} onChange={() => toggleSelected(version.id)} />
                      <span>{checked && <Check size={13} />}</span>
                    </label>
                    <div className="version-primary">
                      <div><strong>{version.name}</strong><span className={`correctness ${version.correctness_status === 'passed' || version.correctness_status === 'valid' ? 'passed' : ''}`}>{version.correctness_status === 'passed' || version.correctness_status === 'valid' ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}{version.correctness_status}</span></div>
                      <p>{version.notes || '没有备注'}</p>
                    </div>
                    <div className="version-metadata"><span>{formatDate(version.created_at)}</span><code>{shortHash(version.source_hash)}</code></div>
                    <div className="version-env"><span>环境</span><code title={run?.environment_fingerprint}>{shortHash(run?.environment_fingerprint ?? run?.environment?.fingerprint)}</code></div>
                    <label className={`baseline-radio ${checked ? '' : 'disabled'}`}>
                      <input type="radio" name="baseline" disabled={!checked} checked={baselineId === version.id} onChange={() => setBaselineId(version.id)} />
                      <span />baseline
                    </label>
                    <div className="row-actions">
                      <button className="icon-button" type="button" aria-label={`编辑 ${version.name}`} onClick={() => openEdit(version)}><Edit3 size={15} /></button>
                      <button className="icon-button danger-icon" type="button" aria-label={`删除 ${version.name}`} onClick={() => setDeleting(version)}><Trash2 size={15} /></button>
                    </div>
                  </article>
                )
              })}
            </div>
          </section>

          {retestJob.job && (
            <section className="panel retest-output"><JobPanel job={retestJob.job} requestError={retestJob.error} onClear={retestJob.clear} /></section>
          )}

          {selected.length < 2 ? (
            <StatusView kind="empty" title="再选择一个版本" description="至少选择两个同题版本后，才会计算逐规模性能比较。" />
          ) : (
            <section className="comparison-section">
              <div className={`comparability-banner ${comparable ? 'comparable' : 'incomparable'}`}>
                {comparable ? <CheckCircle2 size={21} /> : <AlertTriangle size={21} />}
                <div>
                  <strong>{comparable ? '可直接比较' : '不可直接比较'}</strong>
                  <p>{comparable ? '题目修订、suite、输入规模、编译配置与环境指纹一致。' : `${reasons.join('；') || '版本口径不一致'}。不会生成误导性的统一排名或 speedup。`}</p>
                </div>
                {comparison && <span className={`environment-consistency ${comparison.environment_consistent ? 'ok' : ''}`}>{comparison.environment_consistent ? '环境一致' : '环境不一致'}</span>}
              </div>

              <section className="panel metrics-panel">
                <header className="panel-heading">
                  <div><h2><BarChart3 size={18} />逐规模测量</h2><p>主指标为 median；p95 与波动用于判断稳定性。</p></div>
                  {comparisonLoading && <span className="loading-label"><LoaderCircle className="spin" size={14} />正在计算</span>}
                </header>
                {comparisonError ? (
                  <StatusView kind="error" compact title="无法生成比较" description={comparisonError.message} />
                ) : !comparison?.rows?.length && !comparisonLoading ? (
                  <StatusView kind="empty" compact title="没有可用的 benchmark 样本" description="可使用当前统一环境重新测试所选版本。" />
                ) : (
                  <div className="metrics-table-wrap">
                    <table className="metrics-table">
                      <thead>
                        <tr>
                          <th>输入规模</th>
                          {selectedVersions.map((version) => <th key={version.id}><span>{version.name}</span>{version.id === baselineId && <em>baseline</em>}</th>)}
                        </tr>
                      </thead>
                      <tbody>
                        {(comparison?.rows ?? []).map((row) => (
                          <tr key={String(row.size)}>
                            <th><code>{row.size}</code></th>
                            {selectedVersions.map((version) => {
                              const metric = comparisonMetric(row, version.id)
                              return (
                                <td key={version.id}>
                                  {metric ? <div className="metric-cell">
                                    <strong>{formatMetric(metric.median_ms)}</strong>
                                    <span>p95 {formatMetric(metric.p95_ms)}</span>
                                    <span>{metric.cv !== undefined ? `CV ${formatPercent(metric.cv)}` : `MAD ${formatMetric(metric.mad_ms)}`}</span>
                                    <span>{metric.sample_count} 样本</span>
                                    <b className={comparable && typeof metric.speedup === 'number' && metric.speedup > 1 ? 'faster' : ''}>
                                      {comparable && typeof metric.speedup === 'number' ? `${metric.speedup.toFixed(2)}×` : '不计算 speedup'}
                                    </b>
                                  </div> : <span className="missing-metric">无样本</span>}
                                </td>
                              )
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>

              <section className="panel environment-compare-panel">
                <header className="panel-heading"><div><h2>比较口径</h2><p>每次 BenchmarkRun 保留其独立环境快照。</p></div></header>
                <div className="environment-compare-grid">
                  {selectedVersions.map((version) => {
                    const run = latestRun(version)
                    const env = run?.environment
                    const flags = run?.compiler_flags ?? run?.compile_flags ?? version.compile_flags
                    const flagsText = Array.isArray(flags) ? flags.join(' ') : flags
                    const imageDigest = env?.container_digest ?? env?.image_digest
                    return <div key={version.id}><strong>{version.name}</strong><dl><div><dt>题目修订</dt><dd>{version.problem_revision}</dd></div><div><dt>Suite</dt><dd><code>{shortHash(run?.suite_hash)}</code></dd></div><div><dt>协议版本</dt><dd>{run?.protocol_version ?? 'unavailable'}</dd></div><div><dt>编译配置</dt><dd title={flagsText}><code>{flagsText || 'unavailable'}</code></dd></div><div><dt>GPU</dt><dd>{env?.gpu_name ?? env?.gpu ?? 'unavailable'}</dd></div><div><dt>驱动</dt><dd>{env?.driver_version ?? 'unavailable'}</dd></div><div><dt>CUDA / NVCC</dt><dd>{env?.cuda_runtime_version ?? env?.cuda_version ?? '—'} / {env?.nvcc_version ?? '—'}</dd></div><div><dt>镜像摘要</dt><dd title={imageDigest}><code>{shortHash(imageDigest)}</code></dd></div><div><dt>环境指纹</dt><dd><code>{shortHash(run?.environment_fingerprint ?? env?.fingerprint)}</code></dd></div><div><dt>预热 / 样本</dt><dd>{run?.warmup ?? '—'} / {run?.iterations ?? '—'}</dd></div></dl></div>
                  })}
                </div>
              </section>

              <section className="panel diff-panel">
                <header className="panel-heading diff-heading">
                  <div><h2><GitCompareArrows size={18} />代码快照 Diff</h2><p>版本代码不可变；名称和备注可编辑。</p></div>
                  <div className="diff-selects">
                    <label>原始<select value={diffOriginal} onChange={(event) => setDiffOriginal(event.target.value)}>{selectedVersions.map((version) => <option value={version.id} key={version.id}>{version.name}</option>)}</select></label>
                    <Code2 size={16} />
                    <label>修改后<select value={diffModified} onChange={(event) => setDiffModified(event.target.value)}>{selectedVersions.map((version) => <option value={version.id} key={version.id}>{version.name}</option>)}</select></label>
                  </div>
                </header>
                <div className="diff-space"><CodeDiff original={original?.source_code ?? ''} modified={modified?.source_code ?? ''} originalLabel={original?.name} modifiedLabel={modified?.name} /></div>
              </section>
            </section>
          )}
        </>
      )}

      <Modal
        open={!!editing}
        title="编辑版本信息"
        subtitle="只修改名称和备注；源码、环境与 benchmark 快照保持不变。"
        onClose={() => !savingEdit && setEditing(null)}
        footer={<><button className="button ghost" disabled={savingEdit} type="button" onClick={() => setEditing(null)}>取消</button><button className="button primary" disabled={savingEdit || !editName.trim()} type="button" onClick={() => void saveEdit()}>{savingEdit && <LoaderCircle className="spin" size={15} />}保存修改</button></>}
      >
        <label className="field"><span>版本名称 <b>*</b></span><input maxLength={80} value={editName} onChange={(event) => setEditName(event.target.value)} /></label>
        <label className="field"><span>备注 <em>可选</em></span><textarea maxLength={500} rows={4} value={editNotes} onChange={(event) => setEditNotes(event.target.value)} /></label>
      </Modal>

      <Modal
        open={!!deleting}
        title="确认删除性能版本？"
        subtitle="这是第二次确认。删除后，关联的 benchmark 历史也将不可恢复。"
        onClose={() => !deleteBusy && setDeleting(null)}
        footer={<><button className="button ghost" disabled={deleteBusy} type="button" onClick={() => setDeleting(null)}>保留版本</button><button className="button danger" disabled={deleteBusy} type="button" onClick={() => void confirmDelete()}>{deleteBusy ? <LoaderCircle className="spin" size={15} /> : <Trash2 size={15} />}确认删除</button></>}
      >
        <div className="delete-warning"><AlertTriangle size={20} /><p>即将永久删除 <strong>「{deleting?.name}」</strong>。编辑器草稿不会受到影响。</p></div>
      </Modal>

      <Modal
        open={retestOpen}
        title="统一环境重新测试？"
        subtitle="所选版本会依次重新验证并 benchmark，不会修改其代码快照。"
        onClose={() => setRetestOpen(false)}
        footer={<><button className="button ghost" type="button" onClick={() => setRetestOpen(false)}>取消</button><button className="button accent" type="button" onClick={() => void startRetest()}><Play size={15} />开始串行重测</button></>}
      >
        <div className="retest-list">{selectedVersions.map((version) => <div key={version.id}><Check size={14} />{version.name}</div>)}</div>
        <p className="modal-help">GPU Job 会严格串行。只有新测量成功时才追加 BenchmarkRun；不会创建新的 Version。</p>
      </Modal>
    </div>
  )
}
