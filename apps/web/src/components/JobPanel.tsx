import { Check, ChevronRight, CircleAlert, Clock3, LoaderCircle, TerminalSquare, X } from 'lucide-react'
import type { Job, TestCaseResult } from '../domain/types'
import { readableStatus } from '../lib/format'
import { StatusView } from './StatusView'

const activeStatuses = new Set(['queued', 'compiling', 'running', 'validating', 'benchmarking'])

function messageFromError(error: Job['error']): string | undefined {
  if (!error) return undefined
  return typeof error === 'string' ? error : error.message ?? error.code ?? '任务执行失败'
}

function Cases({ cases }: { cases: TestCaseResult[] }) {
  if (!cases.length) return null
  return (
    <div className="case-list">
      {cases.map((item, index) => {
        const passed = item.passed ?? (item.status === 'passed' || item.status === 'succeeded')
        return (
          <div className={`case-row ${passed ? 'passed' : 'failed'}`} key={`${item.name ?? 'case'}-${index}`}>
            <span className="case-icon">{passed ? <Check size={14} /> : <X size={14} />}</span>
            <div>
              <strong>{item.name ?? `公开用例 ${index + 1}`}</strong>
              {(item.message || item.error_type) && <p>{item.error_type ? `${item.error_type} · ` : ''}{item.message}</p>}
            </div>
            {item.duration_ms !== undefined && <time>{item.duration_ms.toFixed(2)} ms</time>}
          </div>
        )
      })}
    </div>
  )
}

export function JobPanel({ job, requestError, onClear }: { job: Job | null; requestError?: Error | null; onClear?: () => void }) {
  if (!job && requestError) {
    return <StatusView kind="error" compact description={requestError.message} />
  }
  if (!job) {
    return (
      <div className="output-idle">
        <div className="output-idle-icon"><TerminalSquare size={20} /></div>
        <strong>等待任务</strong>
        <p>编译诊断、公开用例与验证结果会显示在这里。普通任务不会创建性能版本。</p>
      </div>
    )
  }
  const active = activeStatuses.has(job.status)
  const timedOut = job.status === 'timed_out'
  const failed = ['failed', 'system_error', 'cancelled'].includes(job.status)
  const resultCases = job.result?.test_cases ?? job.result?.cases ?? []
  const errorMessage = messageFromError(job.error)
  const stdout = typeof job.result?.stdout === 'string' ? job.result.stdout : ''
  const stderr = typeof job.result?.stderr === 'string' ? job.result.stderr : ''
  const summary = typeof job.result?.summary === 'string' ? job.result.summary : job.result?.message
  const errorStage = typeof job.error === 'object' && job.error ? job.error.stage : undefined
  const diagnosticsTitle = job.phase === 'compiling' || errorStage === 'compiling' || job.action === 'compile'
    ? (job.language === 'triton_python' ? 'Triton / Python 诊断' : 'NVCC 诊断')
    : '运行输出（已限制）'

  return (
    <div className="job-panel">
      <div className="job-head">
        <div className={`job-state ${active ? 'active' : job.status}`}>
          {active ? <LoaderCircle className="spin" size={16} /> : job.status === 'succeeded' ? <Check size={16} /> : timedOut ? <Clock3 size={16} /> : <CircleAlert size={16} />}
          <strong>{readableStatus(job.status)}</strong>
        </div>
        <span className="job-language">{job.language === 'triton_python' ? 'Triton' : 'CUDA C++'}</span>
        <span className="job-id">任务 {job.id.slice(0, 8)}</span>
        {onClear && !active && <button className="text-button" type="button" onClick={onClear}>清空</button>}
      </div>
      {active && (
        <div className="job-progress-block">
          <div className="job-progress-meta">
            <span>{job.phase ? readableStatus(job.phase) : readableStatus(job.status)}</span>
            <span>{job.queue_position ? `队列第 ${job.queue_position} 位` : job.progress !== undefined ? `${Math.round(job.progress)}%` : '请稍候'}</span>
          </div>
          <div className="progress-track"><span style={{ width: `${job.progress ?? 24}%` }} /></div>
        </div>
      )}
      {timedOut && <StatusView kind="timeout" compact description={errorMessage ?? '运行达到服务端时间限制，临时资源将由后台清理。'} />}
      {failed && <StatusView kind="error" compact title={readableStatus(job.status)} description={errorMessage ?? '任务没有成功完成。'} />}
      {job.status === 'succeeded' && summary && <div className="success-summary"><Check size={15} />{summary}</div>}
      <Cases cases={resultCases} />
      {job.diagnostics && (
        <section className="console-block">
          <header><ChevronRight size={14} /> {diagnosticsTitle}</header>
          <pre>{job.diagnostics}</pre>
        </section>
      )}
      {stderr && (
        <section className="console-block error-stream">
          <header><ChevronRight size={14} /> stderr</header>
          <pre>{stderr}</pre>
        </section>
      )}
      {stdout && (
        <section className="console-block">
          <header><ChevronRight size={14} /> stdout（已限制）</header>
          <pre>{stdout}</pre>
        </section>
      )}
    </div>
  )
}
