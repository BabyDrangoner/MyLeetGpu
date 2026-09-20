import { CheckCircle2, Cloud, Cpu, RefreshCw } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { EnvironmentSnapshot, GpuExecutionTarget, KernelLanguage } from '../domain/types'
import { executionTargetLabel, useExecutionSettings } from '../hooks/useExecutionSettings'
import { languageLabel } from '../lib/languages'

export function ExecutionSettings({ language }: { language: KernelLanguage }) {
  const execution = useExecutionSettings()
  const settings = execution?.settings
  const [candidate, setCandidate] = useState<GpuExecutionTarget | null>(null)
  const [acknowledged, setAcknowledged] = useState(false)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')
  const [saveError, setSaveError] = useState('')
  const [probe, setProbe] = useState<EnvironmentSnapshot | null>(null)
  const [probeError, setProbeError] = useState('')
  const [probing, setProbing] = useState(false)
  const probeSequence = useRef(0)
  const selected = candidate ?? settings?.target ?? 'local'
  const changed = !!settings && (selected !== settings.target || (selected === 'colab' && acknowledged !== settings.colab_acknowledged))

  useEffect(() => {
    if (!settings || candidate !== null) return
    setAcknowledged(settings.colab_acknowledged)
  }, [settings, candidate])

  useEffect(() => {
    ++probeSequence.current
    setProbing(false)
    setProbe(null)
    setProbeError('')
    return () => { ++probeSequence.current }
  }, [selected, language])

  async function testConnection() {
    const request = ++probeSequence.current
    setProbing(true)
    setProbe(null)
    setProbeError('')
    try {
      const result = await api.executionSettings.probe(selected, language)
      if (request === probeSequence.current) setProbe(result)
    } catch (error) {
      if (request === probeSequence.current) setProbeError(error instanceof Error ? error.message : '连接测试失败')
    } finally {
      if (request === probeSequence.current) setProbing(false)
    }
  }

  async function saveSettings() {
    if (!execution || !settings || saving || (selected === 'colab' && !acknowledged)) return
    setSaving(true)
    setMessage('')
    setSaveError('')
    try {
      const result = await execution.save({ target: selected, colab_acknowledged: acknowledged })
      setCandidate(null)
      setAcknowledged(result.colab_acknowledged)
      setMessage(`已保存，后续任务使用 ${executionTargetLabel(result.target)}。已有任务和草稿不受影响。`)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : '设置保存失败')
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="panel execution-settings" aria-labelledby="execution-settings-title">
      <header className="execution-settings-heading">
        <div><h2 id="execution-settings-title">执行位置</h2><p>设置保存到服务端，仅影响之后提交的任务；不会修改代码草稿。</p></div>
        <span className="execution-current" data-testid="active-execution-target">当前：{executionTargetLabel(settings?.target)}{execution?.loading && settings ? ' · 更新中' : ''}</span>
      </header>
      {execution?.error && <div className="execution-feedback" role="alert">无法读取执行设置：{execution.error.message}<button className="text-button" type="button" onClick={() => void execution.reload()}>重试读取设置</button></div>}
      {!settings && !execution?.error && <p role="status">正在读取执行设置…</p>}
      <fieldset className="execution-target-options" disabled={!settings || saving}>
        <legend>选择执行位置</legend>
        <label className={`execution-target-option${selected === 'local' ? ' selected' : ''}`}>
          <input type="radio" name="execution-target" value="local" checked={selected === 'local'} onChange={() => { setCandidate('local'); setMessage(''); setSaveError('') }} />
          <Cpu size={20} aria-hidden="true" /><span><strong>本地 GPU</strong><small>在 Worker 所在机器的受限 Docker 容器中运行。</small></span>
        </label>
        <label className={`execution-target-option${selected === 'colab' ? ' selected' : ''}`}>
          <input type="radio" name="execution-target" value="colab" checked={selected === 'colab'} onChange={() => { setCandidate('colab'); setMessage(''); setSaveError('') }} />
          <Cloud size={20} aria-hidden="true" /><span><strong>Google Colab</strong><small>复用已连接的 Colab GPU，通过 SSH 运行。</small></span>
        </label>
      </fieldset>

      {selected === 'colab' && <div className="execution-colab-details">
        <dl><div><dt>Worker 的 SSH 别名</dt><dd><code>{settings?.colab.ssh_host ?? 'colab-vscode'}</code></dd></div><div><dt>远端工作目录</dt><dd><code>{settings?.colab.remote_root ?? '/content/project/myleetgpu-runner'}</code></dd></div></dl>
        <p>请先在 Worker 所在机器连接现有 Colab，并保持已建立的 SSH ControlMaster。页面仅复用连接，不会创建或停止运行时，也不接收密钥或 Google 登录凭据。Colab 文件系统是临时的，运行时回收或重置后会丢失。</p>
        <label className="execution-consent"><input type="checkbox" checked={acknowledged} disabled={!settings || saving} onChange={(event) => { setCandidate(selected); setAcknowledged(event.target.checked); setMessage('') }} /><span>我理解：Colab 不使用 Docker 隔离，仅适合可信代码。提交任务时会发送代码和测试程序，代码可能访问 Colab 虚拟机中的文件、凭据及其他资源。</span></label>
      </div>}

      <div className="execution-settings-actions">
        <button className="button secondary" type="button" disabled={!settings || probing || saving} onClick={() => void testConnection()}><RefreshCw size={15} className={probing ? 'spin' : undefined} />{probing ? '正在测试连接' : '测试连接'}</button>
        <span className="execution-probe-caption">{executionTargetLabel(selected)} · {languageLabel(language)} · 测试不会切换执行位置</span>
        <span className="execution-save-state">{!settings ? '尚未读取已保存设置' : changed ? '有未保存的更改' : '与已保存设置一致'}</span>
        <button className="button primary" type="button" disabled={!settings || !changed || saving || (selected === 'colab' && !acknowledged)} onClick={() => void saveSettings()}>{saving ? '正在保存…' : '保存设置'}</button>
      </div>
      <div className="execution-feedback" role="status" aria-live="polite">
        {probing && <p>正在等待 Worker 测试 {executionTargetLabel(selected)} 的 {languageLabel(language)} 环境，最多约 3 分钟；不会打断正在执行的任务。</p>}
        {probe && <p className={(probe.healthy ?? probe.status === 'healthy') ? 'execution-probe-success' : 'execution-probe-failure'}>{(probe.healthy ?? probe.status === 'healthy') && <CheckCircle2 size={15} />}<strong>{executionTargetLabel(selected)} · {languageLabel(language)}：{(probe.healthy ?? probe.status === 'healthy') ? '连接正常' : '环境未就绪'}</strong>{probe.gpu_name && <span>{probe.gpu_name}</span>}{probe.message && <span>{probe.message}</span>}</p>}
        {message && <p>{message}</p>}
      </div>
      {(probeError || saveError) && <div className="execution-feedback execution-probe-failure" role="alert">{saveError || probeError}</div>}
    </section>
  )
}
