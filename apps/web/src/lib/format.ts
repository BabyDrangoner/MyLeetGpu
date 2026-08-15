export function formatDate(value?: string): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.valueOf())) return value
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

export function formatMetric(value: number | null | undefined, unit = 'ms'): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  const digits = Math.abs(value) < 0.1 ? 4 : Math.abs(value) < 10 ? 3 : 2
  return `${value.toFixed(digits)}${unit ? ` ${unit}` : ''}`
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(1)}%`
}

export function readableStatus(status?: string): string {
  const labels: Record<string, string> = {
    queued: '排队中',
    compiling: '正在编译',
    running: '正在运行公开样例',
    validating: '正在完整验证',
    benchmarking: '正在基准测试',
    succeeded: '已成功',
    failed: '未通过',
    timed_out: '已超时',
    cancelled: '已取消',
    system_error: '系统错误',
    healthy: '可用',
    unhealthy: '不可用',
  }
  return status ? labels[status] ?? status : '未知'
}

export function difficultyLabel(difficulty: string): string {
  const labels: Record<string, string> = {
    easy: '简单',
    medium: '中等',
    hard: '困难',
    beginner: '入门',
  }
  return labels[difficulty.toLowerCase()] ?? difficulty
}
