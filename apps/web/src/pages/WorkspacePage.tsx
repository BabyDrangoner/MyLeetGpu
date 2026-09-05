import { Archive, Braces, ChevronDown, ChevronLeft, ChevronRight, Clock3, FileCheck2, PanelsTopLeft, PanelLeftClose, PanelLeftOpen, Play, RotateCcw, Save, Settings2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import { CodeEditor } from '../components/CodeEditor'
import { JobPanel } from '../components/JobPanel'
import { MarkdownText } from '../components/MarkdownText'
import { Modal } from '../components/Modal'
import { ResizeHandle } from '../components/ResizeHandle'
import { SaveVersionDialog, type SaveVersionPayload } from '../components/SaveVersionDialog'
import { RetryButton, StatusView } from '../components/StatusView'
import { useToast } from '../components/Toast'
import type { Job, JobAction, KernelLanguage, SavedVersion } from '../domain/types'
import { useAsync } from '../hooks/useAsync'
import { useJob } from '../hooks/useJob'
import { useWorkspaceLayout } from '../hooks/useWorkspaceLayout'
import { readLocalDraft, saveLocalDraft } from '../lib/drafts'
import { difficultyLabel, formatDate, readableStatus } from '../lib/format'
import { implementationLanguages, isKernelLanguage, languageLabel } from '../lib/languages'

const difficultyClass = (difficulty: string) => difficulty.includes('困难') || difficulty === 'hard' ? 'hard' : difficulty.includes('中等') || difficulty === 'medium' ? 'medium' : 'easy'

export function WorkspacePage() {
  const { slug = '' } = useParams()
  const [searchParams, setSearchParams] = useSearchParams()
  const toast = useToast()
  const problem = useAsync(() => api.problems.get(slug), [slug])
  const [source, setSource] = useState('')
  const [loadedDraftKey, setLoadedDraftKey] = useState<string>()
  const [draftSavedAt, setDraftSavedAt] = useState<string>()
  const [draftRemote, setDraftRemote] = useState<'idle' | 'saving' | 'saved' | 'local-only'>('idle')
  const [statementTab, setStatementTab] = useState<'statement' | 'protocol'>('statement')
  const [focusMode, setFocusMode] = useState(false)
  const [outputExpanded, setOutputExpanded] = useState(true)
  const [resetOpen, setResetOpen] = useState(false)
  const [saveOpen, setSaveOpen] = useState(false)
  const [saveSnapshot, setSaveSnapshot] = useState<{ language: KernelLanguage; source: string }>()
  const [versions, setVersions] = useState<SavedVersion[]>([])
  const autosaveSequence = useRef(0)
  const draftLoadSequence = useRef(0)
  const sourceRevision = useRef(0)
  const sourceRef = useRef(source)
  const submissionLock = useRef(false)
  const layout = useWorkspaceLayout()

  const supportedLanguages = useMemo(() => {
    const implementations = problem.data?.implementations
    return implementationLanguages.filter((language) => implementations?.[language])
  }, [problem.data])
  const requestedLanguage = searchParams.get('language')
  const language = isKernelLanguage(requestedLanguage) && supportedLanguages.includes(requestedLanguage)
    ? requestedLanguage
    : problem.data?.default_language ?? supportedLanguages[0] ?? 'cuda_cpp'
  const implementation = problem.data?.implementations[language]
  const draftKey = `${slug}:${language}`
  const draftReady = loadedDraftKey === draftKey

  const refreshVersions = useCallback(async () => {
    try {
      const items = await api.versions.list(slug)
      setVersions(items)
    } catch {
      // Version count is secondary on the coding route.
    }
  }, [slug])

  const onJobSettled = useCallback((settled: Job) => {
    setOutputExpanded(true)
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
    if (jobs.error) setOutputExpanded(true)
  }, [jobs.error])

  useEffect(() => {
    if (!problem.data || searchParams.get('language') === language) return
    const next = new URLSearchParams(searchParams)
    next.set('language', language)
    setSearchParams(next, { replace: true })
  }, [language, problem.data, searchParams, setSearchParams])

  useEffect(() => {
    if (!problem.data || !implementation) return
    const loadSequence = ++draftLoadSequence.current
    ++autosaveSequence.current
    let cancelled = false
    sourceRevision.current = 0
    setLoadedDraftKey(undefined)
    setDraftRemote('idle')
    const local = readLocalDraft(slug, language)
    sourceRef.current = local?.source ?? implementation.starter_code
    setSource(sourceRef.current)
    setDraftSavedAt(local?.updatedAt)
    void api.drafts.get(slug, language).then((remote) => {
      if (cancelled || loadSequence !== draftLoadSequence.current) return
      const localTime = local ? new Date(local.updatedAt).valueOf() : 0
      const remoteTime = remote.updated_at ? new Date(remote.updated_at).valueOf() : 0
      if (sourceRevision.current === 0 && remote.source && remoteTime > localTime) {
        sourceRef.current = remote.source
        setSource(remote.source)
        saveLocalDraft(slug, language, remote.source)
        setDraftSavedAt(remote.updated_at)
      }
      setDraftRemote('saved')
    }).catch((_error) => {
      if (cancelled || loadSequence !== draftLoadSequence.current) return
      setDraftRemote('local-only')
    }).finally(() => {
      if (!cancelled && loadSequence === draftLoadSequence.current) setLoadedDraftKey(draftKey)
    })
    void refreshVersions()
    return () => {
      cancelled = true
      ++autosaveSequence.current
    }
  }, [draftKey, implementation, language, problem.data?.slug, refreshVersions, slug])

  useEffect(() => {
    if (!draftReady || !problem.data) return
    const sequence = ++autosaveSequence.current
    const localTimer = window.setTimeout(() => {
      if (sequence !== autosaveSequence.current) return
      const saved = saveLocalDraft(slug, language, source)
      setDraftSavedAt(saved.updatedAt)
      setDraftRemote('saving')
    }, 350)
    const remoteTimer = window.setTimeout(() => {
      if (sequence !== autosaveSequence.current) return
      void api.drafts.save(slug, language, source).then((draft) => {
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
  }, [draftKey, draftReady, language, problem.data, slug, source])

  const selectLanguage = (nextLanguage: KernelLanguage) => {
    if (jobs.busy || saveOpen || nextLanguage === language) return
    if (draftReady) {
      const saved = saveLocalDraft(slug, language, source)
      setDraftSavedAt(saved.updatedAt)
      void api.drafts.save(slug, language, source).catch(() => undefined)
    }
    ++autosaveSequence.current
    setLoadedDraftKey(undefined)
    jobs.clear()
    const next = new URLSearchParams(searchParams)
    next.set('language', nextLanguage)
    setSearchParams(next)
  }

  const updateSource = useCallback((next: string) => {
    sourceRevision.current += 1
    sourceRef.current = next
    setSource(next)
  }, [])

  const startAction = useCallback(async (action: Exclude<JobAction, 'rebenchmark'>, payload?: SaveVersionPayload) => {
    if (!draftReady || problem.loading || problem.error || resetOpen || jobs.busy || submissionLock.current || (saveOpen && action !== 'save_version')) return
    submissionLock.current = true
    setOutputExpanded(true)
    try {
      await jobs.start({
        problem_id: slug,
        language: payload?.language ?? language,
        action,
        source: payload?.source ?? sourceRef.current,
        version_name: payload?.name,
        notes: payload?.notes,
        allow_duplicate: payload?.allowDuplicate,
      })
      if (action === 'save_version') setSaveOpen(false)
    } catch (error) {
      toast.show(error instanceof Error ? error.message : '提交任务失败', 'error')
    } finally {
      submissionLock.current = false
    }
  }, [draftReady, jobs.busy, jobs.start, language, problem.error, problem.loading, resetOpen, saveOpen, slug, toast])

  useEffect(() => {
    const runSample = (event: KeyboardEvent) => {
      if (event.key !== 'Enter' || !(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey || event.repeat || event.isComposing || event.defaultPrevented) return
      if (!draftReady || problem.loading || problem.error || jobs.busy || saveOpen || resetOpen) return
      event.preventDefault()
      event.stopPropagation()
      void startAction('run')
    }
    window.addEventListener('keydown', runSample, true)
    return () => window.removeEventListener('keydown', runSample, true)
  }, [draftReady, jobs.busy, problem.error, problem.loading, resetOpen, saveOpen, startAction])

  if (problem.loading) return <div className="page"><StatusView kind="loading" title="正在打开编程工作台" /></div>
  if (problem.error || !problem.data) {
    return <div className="page"><StatusView kind="error" description={problem.error?.message ?? '题目不存在。'} action={<RetryButton onClick={() => void problem.reload()} />} /></div>
  }
  const detail = problem.data
  const activeImplementation = implementation ?? detail.implementations[detail.default_language]
  if (!activeImplementation) {
    return <div className="page"><StatusView kind="error" description="题目没有可用的语言实现。" /></div>
  }
  const languageVersionCount = versions.filter((version) => version.language === language).length

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
        <div className="workspace-controls">
          <button className="button secondary" type="button" title="恢复题目、代码与输出的默认比例，不会修改代码" onClick={layout.reset}>
            <PanelsTopLeft size={16} />重置布局
          </button>
          <button className="button secondary" type="button" aria-pressed={focusMode} aria-controls="problem-statement" onClick={() => setFocusMode((focused) => !focused)}>
            {focusMode ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}{focusMode ? '显示题目' : '专注编码'}
          </button>
          <Link className="button secondary" to={`/problems/${encodeURIComponent(slug)}/versions?language=${language}`}>
            <Archive size={16} />性能版本 <span className="count-badge">{languageVersionCount}</span>
          </Link>
        </div>
      </header>

      <div className={`workspace-grid${focusMode ? ' is-focused' : ''}`} ref={layout.workspace.attach} style={layout.workspaceStyle}>
        <section className="statement-panel panel" id="problem-statement" aria-label="题目与测量说明" hidden={focusMode}>
          <div className="tab-bar">
            <button className={statementTab === 'statement' ? 'active' : ''} type="button" aria-pressed={statementTab === 'statement'} onClick={() => setStatementTab('statement')}>题目说明</button>
            <button className={statementTab === 'protocol' ? 'active' : ''} type="button" aria-pressed={statementTab === 'protocol'} onClick={() => setStatementTab('protocol')}>测量协议</button>
          </div>
          <div className="statement-scroll">
            {statementTab === 'statement' ? (
              <>
                <MarkdownText source={detail.statement_markdown} />
                {activeImplementation.instructions_markdown && <MarkdownText source={activeImplementation.instructions_markdown} />}
                {!!detail.constraints?.length && (
                  <section className="constraint-block">
                    <h3>约束</h3>
                    <ul>{detail.constraints.map((constraint) => <li key={constraint}><code>{constraint}</code></li>)}</ul>
                  </section>
                )}
                {activeImplementation.signature && <section className="signature-block"><span>{languageLabel(language)} 实现接口</span><code>{activeImplementation.signature}</code></section>}
              </>
            ) : (
              <div className="protocol-content">
                <div className="protocol-callout">
                  <Clock3 size={20} />
                  <div><strong>平台 harness 计时</strong><p>排除编译或预检查、容器启动、初始化、分配和拷贝；先预热，再用设备事件采样。</p></div>
                </div>
                <dl>
                  <div><dt>输入规模</dt><dd>{detail.benchmark?.input_sizes?.join(' / ') ?? '由题目配置'}</dd></div>
                  <div><dt>预热次数</dt><dd>{detail.benchmark?.warmup ?? '由题目配置'}</dd></div>
                  <div><dt>采样次数</dt><dd>{detail.benchmark?.iterations ?? '由题目配置'}</dd></div>
                  <div><dt>核心指标</dt><dd>median；同时记录 p95、min 与波动</dd></div>
                  <div><dt>可比条件</dt><dd>实现语言、题目修订、suite、规模、执行配置及环境指纹一致</dd></div>
                </dl>
                <p className="muted-copy">温度、功耗策略与后台 GPU 工作会造成波动。本机结果不代表跨机器的绝对排名。</p>
              </div>
            )}
          </div>
        </section>

        {!focusMode && <ResizeHandle axis="x" {...layout.statement} containerRef={layout.workspace.ref} label="调整题目与代码宽度" controls="problem-statement workspace-editor-column" />}

        <section className={`editor-column${outputExpanded ? '' : ' is-output-collapsed'}`} id="workspace-editor-column" aria-label="代码与运行结果" ref={layout.column.attach} style={layout.editorStyle}>
          <div className="editor-panel panel" id="workspace-editor">
            <div className="editor-toolbar">
              <div className="file-label"><Braces size={16} /><strong>solution{activeImplementation.file_extension}</strong><span>{activeImplementation.display_name}</span></div>
              <div className="language-switch" role="group" aria-label="实现语言">
                {supportedLanguages.map((item) => (
                  <button key={item} className={language === item ? 'active' : ''} type="button" aria-pressed={language === item} disabled={jobs.busy || saveOpen || resetOpen} onClick={() => selectLanguage(item)}>{languageLabel(item)}</button>
                ))}
              </div>
              <div className="editor-tool-actions">
                <div className="draft-state" title={draftSavedAt ? `保存于 ${formatDate(draftSavedAt)}` : undefined}>
                  <span className={draftRemote === 'saving' ? 'pulse-dot' : 'saved-dot'} />
                  {!draftReady ? '正在读取草稿' : draftRemote === 'saving' ? '正在保存' : draftRemote === 'local-only' ? '已本地保存' : '草稿已保存'}
                </div>
                <button className="text-button" type="button" disabled={!draftReady || jobs.busy || saveOpen} onClick={() => setResetOpen(true)}><RotateCcw size={14} />重置</button>
              </div>
            </div>
            <div className="editor-space"><CodeEditor value={source} language={language} problemId={slug} readOnly={!draftReady} onChange={updateSource} /></div>
            <div className="action-bar">
              <div className="action-group">
                <button className="button primary" disabled={jobs.busy || !draftReady} type="button" aria-keyshortcuts="Control+Enter Meta+Enter" title="运行样例（Ctrl / ⌘ + Enter）" onClick={() => void startAction('run')}><Play size={16} />运行样例<kbd className="shortcut-hint" aria-hidden="true">Ctrl/⌘ ↵</kbd></button>
                <button className="button secondary" disabled={jobs.busy || !draftReady} type="button" onClick={() => void startAction('compile')}><Settings2 size={16} />{language === 'torch_python' ? '代码检查' : '编译'}</button>
                <button className="button secondary" disabled={jobs.busy || !draftReady} type="button" onClick={() => void startAction('validate')}><FileCheck2 size={16} />完整验证</button>
              </div>
              <button
                className="button secondary"
                disabled={jobs.busy || !draftReady}
                type="button"
                onClick={() => { setSaveSnapshot({ language, source }); setSaveOpen(true) }}
              ><Save size={16} />保存为性能版本</button>
            </div>
          </div>
          {outputExpanded && <ResizeHandle axis="y" {...layout.editor} containerRef={layout.column.ref} label="调整代码与输出高度" controls="workspace-editor workspace-output-panel" />}
          <section className={`output-panel panel${outputExpanded ? '' : ' is-collapsed'}`} id="workspace-output-panel" aria-label="任务与输出">
            <header className="output-header">
              <button className="output-toggle" type="button" aria-label={outputExpanded ? '收起任务与输出' : '展开任务与输出'} aria-expanded={outputExpanded} aria-controls="workspace-output" onClick={() => setOutputExpanded((expanded) => !expanded)}>
                {outputExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}<strong>任务与输出</strong>
              </button>
              <span className="output-status" role="status">{jobs.submitting ? '正在提交' : jobs.error ? '请求失败' : jobs.job ? (outputExpanded ? '' : readableStatus(jobs.job.status)) : '尚未运行'}</span>
            </header>
            <div className="output-scroll" id="workspace-output" hidden={!outputExpanded}><JobPanel job={jobs.job} requestError={jobs.error} onClear={jobs.clear} /></div>
          </section>
        </section>
      </div>

      <Modal
        open={resetOpen}
        title="重置为 Starter Code？"
        subtitle="当前草稿会被 starter 覆盖；手动保存的性能版本不受影响。"
        onClose={() => setResetOpen(false)}
        footer={<><button className="button ghost" type="button" onClick={() => setResetOpen(false)}>取消</button><button className="button danger" type="button" onClick={() => { updateSource(activeImplementation.starter_code); setResetOpen(false); toast.show(`已恢复 ${languageLabel(language)} Starter Code`, 'info') }}><RotateCcw size={16} />确认重置</button></>}
      >
        <div className="inline-notice warning">重置后仍会自动保存为本题草稿。此操作不会运行、验证或创建版本。</div>
      </Modal>
      <SaveVersionDialog
        open={saveOpen}
        problemId={slug}
        language={saveSnapshot?.language ?? language}
        snapshot={saveSnapshot?.source ?? source}
        currentSource={source}
        existingVersions={versions}
        busy={jobs.submitting}
        onClose={() => setSaveOpen(false)}
        onSave={(payload) => void startAction('save_version', payload)}
      />
    </div>
  )
}
