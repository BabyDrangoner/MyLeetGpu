import { RefreshCw } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { EnvironmentSnapshot, KernelLanguage } from '../domain/types'
import { languageLabel } from '../lib/languages'

export function CpuExecutionStatus({ language, onProbed }: { language: KernelLanguage; onProbed: () => Promise<unknown> }) {
  const [probing, setProbing] = useState(false)
  const [result, setResult] = useState<EnvironmentSnapshot | null>(null)
  const [error, setError] = useState('')
  const sequence = useRef(0)

  useEffect(() => {
    ++sequence.current
    setProbing(false)
    setResult(null)
    setError('')
    return () => { ++sequence.current }
  }, [language])

  async function probe() {
    const request = ++sequence.current
    setProbing(true)
    setResult(null)
    setError('')
    try {
      const snapshot = await api.executionSettings.probe('cpu', language)
      if (request !== sequence.current) return
      setResult(snapshot)
      await onProbed()
    } catch (cause) {
      if (request === sequence.current) setError(cause instanceof Error ? cause.message : 'CPU 环境探测失败')
    } finally {
      if (request === sequence.current) setProbing(false)
    }
  }

  return (
    <section className="panel execution-settings" aria-labelledby="cpu-execution-title">
      <header className="execution-settings-heading">
        <div><h2 id="cpu-execution-title">CPU 执行环境</h2><p>C++ / Python 题固定在 Worker 所在机器的 CPU 上执行，不依赖本地 GPU 或 Colab，不改变 GPU 执行设置。</p></div>
        <span className="execution-current">当前：本地 CPU</span>
      </header>
      <p>使用本机原生进程，并非安全沙箱；代码可访问当前用户可用的文件和资源，仅运行可信代码。</p>
      <div className="execution-settings-actions">
        <button className="button secondary" type="button" disabled={probing} onClick={() => void probe()}><RefreshCw size={15} className={probing ? 'spin' : undefined} />{probing ? '正在探测 CPU' : '探测 CPU 环境'}</button>
        <span className="execution-probe-caption">{languageLabel(language)} · {language === 'cpp' ? 'C++17 编译器' : 'Python 标准库'}</span>
      </div>
      <div className="execution-feedback" role="status" aria-live="polite">
        {probing && <p>正在等待 Worker 探测 CPU 环境。</p>}
        {result && <p>{languageLabel(language)}：{result.healthy ? '环境就绪' : '环境未就绪'}{result.message ? ` · ${result.message}` : ''}</p>}
      </div>
      {error && <div className="execution-feedback execution-probe-failure" role="alert">{error}</div>}
    </section>
  )
}
