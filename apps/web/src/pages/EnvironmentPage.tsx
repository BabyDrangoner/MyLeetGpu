import { AlertTriangle, Box, CheckCircle2, Cpu, Fingerprint, Gauge, RefreshCw, ShieldCheck, TerminalSquare } from 'lucide-react'
import { api } from '../api/client'
import { RetryButton, StatusView } from '../components/StatusView'
import { useAsync } from '../hooks/useAsync'
import { formatDate } from '../lib/format'

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
  const environment = useAsync(() => api.environment(), [])
  if (environment.loading) return <div className="page"><StatusView kind="loading" title="正在探测 GPU 与 CUDA 环境" description="检查本地 Runner 报告的最近一次环境快照。" /></div>
  if (environment.error || !environment.data) {
    return <div className="page"><StatusView kind="error" title="无法读取环境状态" description={environment.error?.message} action={<RetryButton onClick={() => void environment.reload()} />} /></div>
  }
  const data = environment.data
  const healthy = data.healthy ?? data.status === 'healthy'
  return (
    <div className="page environment-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">LOCAL RUNTIME</div>
          <h1>运行环境</h1>
          <p>这里展示 Runner 最近一次真实探测结果；缺失指标会诚实标记为 unavailable。</p>
        </div>
        <button className="button secondary" type="button" onClick={() => void environment.reload()}><RefreshCw size={16} />刷新状态</button>
      </div>

      <section className={`environment-hero ${healthy ? 'healthy' : 'unhealthy'}`}>
        <div className="environment-device-mark"><Cpu size={37} /></div>
        <div className="environment-device-copy">
          <span className="environment-kicker">NVIDIA GPU</span>
          <h2>{data.gpu_name ?? data.gpu ?? '未检测到 GPU'}</h2>
          <p>{data.message ?? (healthy ? 'CUDA 执行链路已通过最近一次健康检查。' : 'Runner 当前不会接受新的 GPU 任务，请按诊断提示恢复。')}</p>
        </div>
        <div className={`health-badge ${healthy ? 'healthy' : 'unhealthy'}`}>
          {healthy ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
          {healthy ? '环境就绪' : '环境异常'}
        </div>
      </section>

      <div className="environment-facts-grid">
        <Fact label="Compute Capability" value={data.compute_capability ? `sm_${data.compute_capability.replace('.', '')}` : undefined} icon={<Gauge size={18} />} />
        <Fact label="Windows 驱动" value={data.driver_version} icon={<TerminalSquare size={18} />} />
        <Fact label="CUDA Runtime" value={data.cuda_runtime_version ?? data.cuda_version} icon={<Cpu size={18} />} />
        <Fact label="NVCC" value={data.nvcc_version} icon={<TerminalSquare size={18} />} />
        <Fact label="容器镜像" value={data.container_image} icon={<Box size={18} />} />
        <Fact label="环境指纹" value={data.fingerprint} icon={<Fingerprint size={18} />} />
      </div>

      <div className="environment-lower-grid">
        <section className="panel environment-details">
          <header><h2>可复现性记录</h2><span>探测于 {formatDate(data.checked_at)}</span></header>
          <dl>
            <div><dt>容器摘要</dt><dd><code>{data.container_digest || 'unavailable'}</code></dd></div>
            <div><dt>环境指纹</dt><dd><code>{data.fingerprint || 'unavailable'}</code></dd></div>
            <div><dt>温度 / 时钟 / GPU busy</dt><dd>{data.unavailable_metrics?.length ? `unavailable：${data.unavailable_metrics.join('、')}` : '以 benchmark 记录为准'}</dd></div>
          </dl>
        </section>
        <section className="panel safety-card">
          <div className="safety-card-icon"><ShieldCheck size={23} /></div>
          <div>
            <h2>可信单机边界</h2>
            <p>用户代码在受限的一次性容器中执行，但消费级 GPU 与 Docker 不提供公网多租户所需的强 GPU / 显存隔离。</p>
            <strong>局域网模式仅限认证后的可信设备；严禁暴露到公网。</strong>
          </div>
        </section>
      </div>
    </div>
  )
}
