import { AlertTriangle, Inbox, LoaderCircle, RotateCcw, TimerOff } from 'lucide-react'
import type { ReactNode } from 'react'

type Kind = 'loading' | 'empty' | 'error' | 'timeout'

const defaults: Record<Kind, { title: string; description: string }> = {
  loading: { title: '正在加载', description: '正在从本地服务读取数据…' },
  empty: { title: '这里还是空的', description: '暂时没有可展示的数据。' },
  error: { title: '加载失败', description: '本地服务返回了错误，请确认 API 已启动。' },
  timeout: { title: '等待超时', description: '任务耗时超过限制，已停止等待结果。' },
}

export function StatusView({
  kind,
  title,
  description,
  action,
  compact = false,
}: {
  kind: Kind
  title?: string
  description?: string
  action?: ReactNode
  compact?: boolean
}) {
  const Icon = kind === 'loading' ? LoaderCircle : kind === 'empty' ? Inbox : kind === 'timeout' ? TimerOff : AlertTriangle
  return (
    <div className={`status-view ${compact ? 'compact' : ''}`} role={kind === 'error' || kind === 'timeout' ? 'alert' : 'status'}>
      <div className={`status-icon ${kind}`}><Icon size={compact ? 20 : 25} /></div>
      <div>
        <strong>{title ?? defaults[kind].title}</strong>
        <p>{description ?? defaults[kind].description}</p>
        {action && <div className="status-action">{action}</div>}
      </div>
    </div>
  )
}

export function RetryButton({ onClick, label = '重新加载' }: { onClick: () => void; label?: string }) {
  return <button className="button secondary small" type="button" onClick={onClick}><RotateCcw size={14} />{label}</button>
}
