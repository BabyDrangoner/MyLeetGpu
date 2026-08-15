import { Archive, Braces, ChevronLeft, Clock3, FileCheck2, Play, RotateCcw, Save, Settings2 } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ApiError, api } from '../api/client'
import { CodeEditor } from '../components/CodeEditor'
import { JobPanel } from '../components/JobPanel'
import { MarkdownText } from '../components/MarkdownText'
import { Modal } from '../components/Modal'
import { SaveVersionDialog, type SaveVersionPayload } from '../components/SaveVersionDialog'
import { RetryButton, StatusView } from '../components/StatusView'
import { useToast } from '../components/Toast'
import type { Job, JobAction, SavedVersion } from '../domain/types'
import { useAsync } from '../hooks/useAsync'
import { useJob } from '../hooks/useJob'
import { readLocalDraft, saveLocalDraft } from '../lib/drafts'
import { difficultyLabel, formatDate } from '../lib/format'

const difficultyClass = (difficulty: string) => difficulty.includes('困难') || difficulty === 'hard' ? 'hard' : difficulty.includes('中等') || difficulty === 'medium' ? 'medium' : 'easy'

export function WorkspacePage() {
  const { slug = '' } = useParams()
  const toast = useToast()
  const problem = useAsync(() => api.problems.get(slug), [slug])
  const [source, setSource] = useState('')
  const [draftReady, setDraftReady] = useState(false)
  const [draftSavedAt, setDraftSavedAt] = useState<string>()
  const [draftRemote, setDraftRemote] = useState<'idle' | 'saving' | 'saved' | 'local-only'>('idle')
  const [statementTab, setStatementTab] = useState<'statement' | 'protocol'>('statement')
  const [resetOpen, setResetOpen] = useState(false)
  const [saveOpen, setSaveOpen] = useState(false)
  const [saveSnapshot, setSaveSnapshot] = useState('')
  const [versions, setVersions] = useState<SavedVersion[]>([])
  const autosaveSequence = useRef(0)

  const refreshVersions = useCallback(async () => {
    try {
      const items = await api.versions.list(slug)
      setVersions(items)
    } catch {
      // Version count is secondary on the coding route.
    }
  }, [slug])

  const onJobSettled = useCallback((settled: Job) => {
    if (settled.status === 'succeeded') {
      const label = settled.action === 'save_version' ? '性能版本已保存' : '任务已完成'
      toast.show(label, 'success')
      if (settled.action === 'save_version') void refreshVersions()
    } else {
      toast.show(settled.status === 'timed_out' ? '任务已超时' : '任务未通过，请查看输出', 'error')
    }
  }, [refreshVersions, toast])
  const jobs = useJob(onJobSettled)

  useEffect(() => {
    if (!problem.data) return
    let cancelled = false
    const local = readLocalDraft(slug)
    setSource(local?.source ?? problem.data.starter_code)
    setDraftSavedAt(local?.updatedAt)
    setDraftReady(true)
    void api.drafts.get(slug).then((remote) => {
      if (cancelled) return
      const localTime = local ? new Date(local.updatedAt).valueOf() : 0
      const remoteTime = remote.updated_at ? new Date(remote.updated_at).valueOf() : 0
      if (remote.source && remoteTime > localTime) {
        setSource(remote.source)
        saveLocalDraft(slug, remote.source)
        setDraftSavedAt(remote.updated_at)
      }
      setDraftRemote('saved')
    }).catch((error) => {
      if (error instanceof ApiError && error.status === 404) setDraftRemote('local-only')
      else setDraftRemote('local-only')
    })
    void refreshVersions()
    return () => { cancelled = true }
  }, [problem.data?.slug, refreshVersions, slug])

  useEffect(() => {
    if (!draftReady || !problem.data) return
    const sequence = ++autosaveSequence.current
    const localTimer = window.setTimeout(() => {
      if (sequence !== autosaveSequence.current) return
      const saved = saveLocalDraft(slug, source)
      setDraftSavedAt(saved.updatedAt)
      setDraftRemote('saving')
    }, 350)
    const remoteTimer = window.setTimeout(() => {
      if (sequence !== autosaveSequence.current) return
      void api.drafts.save(slug, source).then((draft) => {
        if (sequence !== autosaveSequence.current) return
        setDraftSavedAt(draft.updated_at)
        setDraftRemote('saved')
      }).catch(() => {
        if (sequence === autosaveSequence.current) setDraftRemote('local-only')
      })
    }, 1_100)
    return () => {
      window.clearTimeout(localTimer)
      window.clearTimeout(remoteTimer)
    }
  }, [draftReady, problem.data, slug, source])

  const startAction = async (action: Exclude<JobAction, 'rebenchmark'>, payload?: SaveVersionPayload) => {
    try {
      await jobs.start({
        problem_id: slug,
        action,
        source: payload?.source ?? source,
        version_name: payload?.name,
        notes: payload?.notes,
        allow_duplicate: payload?.allowDuplicate,
      })
      if (action === 'save_version') setSaveOpen(false)
    } catch (error) {
      toast.show(error instanceof Error ? error.message : '提交任务失败', 'error')
    }
  }

  if (problem.loading) return <div className="page"><StatusView kind="loading" title="正在打开 CUDA 工作台" /></div>
  if (problem.error || !problem.data) {
    return <div className="page"><StatusView kind="error" description={problem.error?.message ?? '题目不存在。'} action={<RetryButton onClick={() => void problem.reload()} />} /></div>
  }
  const detail = problem.data

  return (
    <div className="workspace-page">
      <header className="workspace-titlebar">
        <div>
          <Link className="back-link" to="/problems"><ChevronLeft size={15} /> 所有题目</Link>
          <div className="workspace-title-row">
            <h1>{detail.title}</h1>
            <span className={`difficulty ${difficultyClass(detail.difficulty)}`}>{difficultyLabel(detail.difficulty)}</span>
            <span className="revision-badge">rev {detail.revision}</span>
          </div>
        </div>
        <Link className="button secondary" to={`/problems/${encodeURIComponent(slug)}/versions`}>
          <Archive size={16} />性能版本 <span className="count-badge">{versions.length}</span>
        </Link>
      </header>

      <div className="workspace-grid">
        <section className="statement-panel panel">
          <div className="tab-bar">
            <button className={statementTab === 'statement' ? 'active' : ''} type="button" onClick={() => setStatementTab('statement')}>题目说明</button>
            <button className={statementTab === 'protocol' ? 'active' : ''} type="button" onClick={() => setStatementTab('protocol')}>测量协议</button>
          </div>
          <div className="statement-scroll">
            {statementTab === 'statement' ? (
              <>
                <MarkdownText source={detail.statement_markdown} />
                {!!detail.constraints?.length && (
                  <section className="constraint-block">
                    <h3>约束</h3>
                    <ul>{detail.constraints.map((constraint) => <li key={constraint}><code>{constraint}</code></li>)}</ul>
                  </section>
                )}
                {detail.signature && <section className="signature-block"><span>实现接口</span><code>{detail.signature}</code></section>}
              </>
            ) : (
              <div className="protocol-content">
                <div className="protocol-callout">
                  <Clock3 size={20} />
                  <div><strong>平台 harness 计时</strong><p>排除编译、容器启动、初始化、分配和拷贝；先预热，再用 CUDA Events 采样。</p></div>
                </div>
                <dl>
                  <div><dt>输入规模</dt><dd>{detail.benchmark?.input_sizes?.join(' / ') ?? '由题目配置'}</dd></div>
                  <div><dt>预热次数</dt><dd>{detail.benchmark?.warmup ?? '由题目配置'}</dd></div>
                  <div><dt>采样次数</dt><dd>{detail.benchmark?.iterations ?? '由题目配置'}</dd></div>
                  <div><dt>核心指标</dt><dd>median；同时记录 p95、min 与波动</dd></div>
                  <div><dt>可比条件</dt><dd>题目修订、suite、规模、编译配置及环境指纹一致</dd></div>
                </dl>
                <p className="muted-copy">温度、功耗策略与后台 GPU 工作会造成波动。本机结果不代表跨机器的绝对排名。</p>
              </div>
            )}
          </div>
        </section>

        <section className="editor-column">
          <div className="editor-panel panel">
            <div className="editor-toolbar">
              <div className="file-label"><Braces size={16} /><strong>solution.cu</strong><span>CUDA C++</span></div>
              <div className="draft-state" title={draftSavedAt ? `保存于 ${formatDate(draftSavedAt)}` : undefined}>
                <span className={draftRemote === 'saving' ? 'pulse-dot' : 'saved-dot'} />
                {draftRemote === 'saving' ? '正在保存' : draftRemote === 'local-only' ? '已本地保存' : '草稿已保存'}
              </div>
              <button className="text-button" type="button" onClick={() => setResetOpen(true)}><RotateCcw size={14} />重置</button>
            </div>
            <div className="editor-space"><CodeEditor value={source} onChange={setSource} /></div>
            <div className="action-bar">
              <div className="action-group">
                <button className="button secondary" disabled={jobs.busy} type="button" onClick={() => void startAction('compile')}><Settings2 size={16} />编译</button>
                <button className="button secondary" disabled={jobs.busy} type="button" onClick={() => void startAction('run')}><Play size={16} />运行样例</button>
                <button className="button secondary" disabled={jobs.busy} type="button" onClick={() => void startAction('validate')}><FileCheck2 size={16} />完整验证</button>
              </div>
              <button
                className="button accent"
                disabled={jobs.busy}
                type="button"
                onClick={() => { setSaveSnapshot(source); setSaveOpen(true) }}
              ><Save size={16} />保存为性能版本</button>
            </div>
          </div>
          <section className="output-panel panel">
            <header className="output-header"><strong>任务与输出</strong><span>输出经过清理与截断</span></header>
            <div className="output-scroll"><JobPanel job={jobs.job} requestError={jobs.error} onClear={jobs.clear} /></div>
          </section>
        </section>
      </div>

      <Modal
        open={resetOpen}
        title="重置为 Starter Code？"
        subtitle="当前草稿会被 starter 覆盖；手动保存的性能版本不受影响。"
        onClose={() => setResetOpen(false)}
        footer={<><button className="button ghost" type="button" onClick={() => setResetOpen(false)}>取消</button><button className="button danger" type="button" onClick={() => { setSource(detail.starter_code); setResetOpen(false); toast.show('已恢复 Starter Code', 'info') }}><RotateCcw size={16} />确认重置</button></>}
      >
        <div className="inline-notice warning">重置后仍会自动保存为本题草稿。此操作不会运行、验证或创建版本。</div>
      </Modal>
      <SaveVersionDialog
        open={saveOpen}
        problemId={slug}
        snapshot={saveSnapshot}
        currentSource={source}
        existingVersions={versions}
        busy={jobs.submitting}
        onClose={() => setSaveOpen(false)}
        onSave={(payload) => void startAction('save_version', payload)}
      />
    </div>
  )
}
