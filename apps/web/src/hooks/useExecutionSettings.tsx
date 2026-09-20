import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useLocation } from 'react-router-dom'
import { api } from '../api/client'
import type { ExecutionSettings, ExecutionTarget } from '../domain/types'

export const executionTargetLabel = (target?: ExecutionTarget) => target === 'cpu' ? '本地 CPU' : target === 'colab' ? 'Colab' : target === 'local' ? '本地 GPU' : '目标未加载'

interface ExecutionSettingsState {
  settings: ExecutionSettings | null
  loading: boolean
  error: Error | null
  reload: () => Promise<void>
  save: (input: Pick<ExecutionSettings, 'target' | 'colab_acknowledged'>) => Promise<ExecutionSettings>
}

const ExecutionSettingsContext = createContext<ExecutionSettingsState | null>(null)

export function ExecutionSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<ExecutionSettings | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const sequence = useRef(0)
  const { pathname } = useLocation()

  const reload = useCallback(async () => {
    const request = ++sequence.current
    setLoading(true)
    setError(null)
    try {
      const result = await api.executionSettings.get()
      if (request === sequence.current) setSettings(result)
    } catch (cause) {
      if (request === sequence.current) setError(cause instanceof Error ? cause : new Error('无法读取执行设置'))
    } finally {
      if (request === sequence.current) setLoading(false)
    }
  }, [])

  const save = useCallback(async (input: Pick<ExecutionSettings, 'target' | 'colab_acknowledged'>) => {
    // A settings read started before this write must not restore the old target.
    const request = ++sequence.current
    let result: ExecutionSettings
    try {
      result = await api.executionSettings.save(input)
    } catch (cause) {
      if (request === sequence.current) setLoading(false)
      throw cause
    }
    if (request === sequence.current) {
      setSettings(result)
      setLoading(false)
      setError(null)
    } else {
      await reload()
    }
    return result
  }, [reload])

  useEffect(() => {
    void reload()
    return () => { ++sequence.current }
  }, [pathname, reload])

  useEffect(() => {
    const refreshVisible = () => { if (document.visibilityState === 'visible') void reload() }
    window.addEventListener('focus', refreshVisible)
    document.addEventListener('visibilitychange', refreshVisible)
    return () => {
      window.removeEventListener('focus', refreshVisible)
      document.removeEventListener('visibilitychange', refreshVisible)
    }
  }, [reload])

  const value = useMemo(() => ({ settings, loading, error, reload, save }), [settings, loading, error, reload, save])
  return <ExecutionSettingsContext.Provider value={value}>{children}</ExecutionSettingsContext.Provider>
}

export function useExecutionSettings() {
  return useContext(ExecutionSettingsContext)
}
