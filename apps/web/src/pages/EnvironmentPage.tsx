import { AlertTriangle, Box, CheckCircle2, Cpu, Fingerprint, Gauge, RefreshCw, ShieldCheck, TerminalSquare } from 'lucide-react'
import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import { RetryButton, StatusView } from '../components/StatusView'
import { ExecutionSettings } from '../components/ExecutionSettings'
import { CpuExecutionStatus } from '../components/CpuExecutionStatus'
import { useAsync } from '../hooks/useAsync'
import type { KernelLanguage } from '../domain/types'
import { formatDate } from '../lib/format'
import { implementationLanguages, isCpuLanguage, isKernelLanguage, languageLabel } from '../lib/languages'
import { executionTargetLabel, useExecutionSettings } from '../hooks/useExecutionSettings'

function Fact({ label, value, icon }: { label: string; value?: string; icon: React.ReactNode }) {
  return (
    <div className="environment-fact">
      <div className="environment-fact-icon">{icon}</div>
      <span>{label}</span>
      <strong title={value}>{value || 'unavailable'}</strong>
    </div>
  )
}

export function EnvironmentPage() {
  const [searchParams] = useSearchParams()
  const requestedLanguage = searchParams.get('language')
  const [language, setLanguage] = useState<KernelLanguage>(isKernelLanguage(requestedLanguage) ? requestedLanguage : 'cuda_cpp')
  const execution = useExecutionSettings()
  const cpu = isCpuLanguage(language)
  const environment = useAsync(() => api.environment(language), [language, cpu ? 'cpu' : execution?.settings?.target ?? 'local', cpu ? null : execution?.settings?.updated_at ?? null])
  const runtimeLabel = languageLabel(language)
  const data = environment.data
  const healthy = data?.healthy ?? data?.status === 'healthy'
  return (
    <div className="page environment-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">EXECUTION & RUNTIME</div>
          <h1>运行环境</h1>
          <p>选择任务的执行位置，并查看对应 Runner 最近一次真实环境快照。</p>
        </div>
        <div className="environment-heading-actions">
          <div className="language-switch" role="group" aria-label="运行环境语言">
            {implementationLanguages.map((item) => (
              <button className={language === item ? 'active' : ''} key={item} type="button" aria-pressed={language === item} onClick={() => setLanguage(item)}>{languageLabel(item)}</button>
            ))}
          </div>
          <button className="button secondary" type="button" disabled={environment.loading} onClick={() => void environment.reload()}><RefreshCw className={environment.loading ? 'spin' : undefined} size={16} />刷新状态</button>
        </div>
      </div>

      {cpu ? <CpuExecutionStatus language={language} onProbed={environment.reload} /> : <ExecutionSettings language={language} />}

      {environment.loading ? (
        <StatusView kind="loading" title={`正在探测 ${runtimeLabel} 运行环境`} description="正在读取当前执行位置最近一次环境快照。" />
      ) : environment.error || !data ? (
        <StatusView kind="error" title="无法读取环境状态" description={environment.error?.message} action={<RetryButton onClick={() => void environment.reload()} />} />
      ) : (
        <>
          <section className={`environment-hero ${healthy ? 'healthy' : 'unhealthy'}`}>
            <div className="environment-device-mark"><Cpu size={37} /></div>
            <div className="environment-device-copy">
              <span className="environment-kicker">{executionTargetLabel(cpu ? 'cpu' : data.execution_target ?? execution?.settings?.target)} · {runtimeLabel} · {cpu ? '固定 CPU 执行' : '已保存的执行位置'}</span>
              <h2>{cpu ? data.cpu_name ?? 'Worker 本机 CPU' : data.gpu_name ?? data.gpu ?? '运行设备信息暂不可用'}</h2>
              <p>{data.message ?? (healthy ? `${runtimeLabel} 执行链路已通过最近一次健康检查。` : '当前运行环境不可用，请按诊断提示恢复。')}</p>
            </div>
            <div className={`health-badge ${healthy ? 'healthy' : 'unhealthy'}`}>
              {healthy ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
              {healthy ? '环境就绪' : '环境异常'}
            </div>
          </section>

          <div className="environment-facts-grid">
            {cpu ? <>
              <Fact label="操作系统" value={data.platform} icon={<TerminalSquare size={18} />} />
              <Fact label="处理器架构" value={data.architecture} icon={<Cpu size={18} />} />
              {language === 'cpp' ? <Fact label="C++17 编译器" value={data.compiler_version} icon={<TerminalSquare size={18} />} /> : <Fact label="Python 标准库" value={data.python_version} icon={<TerminalSquare size={18} />} />}
            </> : <>
              <Fact label="Compute Capability" value={data.compute_capability ? `sm_${data.compute_capability.replace('.', '')}` : undefined} icon={<Gauge size={18} />} />
              <Fact label="驱动" value={data.driver_version} icon={<TerminalSquare size={18} />} />
              <Fact label="CUDA Runtime" value={data.cuda_runtime_version ?? data.cuda_version} icon={<Cpu size={18} />} />
            </>}
            {language === 'cuda_cpp' && <Fact label="NVCC" value={data.nvcc_version} icon={<TerminalSquare size={18} />} />}
            {language === 'triton_python' && <Fact label="Python / PyTorch / Triton" value={[data.python_version, data.torch_version, data.triton_version].filter(Boolean).join(' / ')} icon={<TerminalSquare size={18} />} />}
            {language === 'torch_python' && <Fact label="Python / PyTorch / Torch CUDA" value={[data.python_version, data.torch_version, data.torch_cuda_version ?? data.cuda_runtime_version ?? data.cuda_version].filter(Boolean).join(' / ')} icon={<TerminalSquare size={18} />} />}
            <Fact label="执行方式" value={cpu ? '可信代码 · 本机原生进程（无沙箱）' : data.execution_target === 'colab' ? '可信代码 · Colab 原生进程（无 Docker）' : data.container_image} icon={<Box size={18} />} />
            <Fact label="环境指纹" value={data.fingerprint} icon={<Fingerprint size={18} />} />
          </div>

          <div className="environment-lower-grid">
            <section className="panel environment-details">
              <header><h2>可复现性记录</h2><span>探测于 {formatDate(data.checked_at)}</span></header>
              <dl>
                {!cpu && <div><dt>容器摘要</dt><dd><code>{data.container_digest || 'unavailable'}</code></dd></div>}
                <div><dt>环境指纹</dt><dd><code>{data.fingerprint || 'unavailable'}</code></dd></div>
                {cpu ? <div><dt>计时方式</dt><dd>CPU 单调时钟；系统负载与频率会影响结果</dd></div> : <div><dt>温度 / 时钟 / GPU busy</dt><dd>{data.unavailable_metrics?.length ? `unavailable：${data.unavailable_metrics.join('、')}` : '以 benchmark 记录为准'}</dd></div>}
              </dl>
            </section>
            <section className="panel safety-card">
              <div className="safety-card-icon"><ShieldCheck size={23} /></div>
              <div>
                <h2>{cpu ? 'CPU 可信代码边界' : data.execution_target === 'colab' ? 'Colab 可信代码边界' : '可信单机边界'}</h2>
                <p>{cpu ? '普通 C++ / Python 在 Worker 机器原生执行，超时控制不等于安全隔离。代码可能读取、修改文件或访问网络；不要运行不可信来源的代码。' : data.execution_target === 'colab' ? '用户代码在 Colab 原生进程中执行，不使用 Docker；仅运行可信代码，不要在同一运行时保留敏感凭据或文件。' : '用户代码在受限的一次性容器中执行，但消费级 GPU 与 Docker 不提供公网多租户所需的强 GPU / 显存隔离。'}</p>
                <strong>局域网模式仅限认证后的可信设备；严禁暴露到公网。</strong>
              </div>
            </section>
          </div>
        </>
      )}
    </div>
  )
}
