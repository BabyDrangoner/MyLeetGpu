import { useCallback, useEffect, useRef, useState } from 'react'

export interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: Error | null
}

export function useAsync<T>(loader: () => Promise<T>, dependencies: unknown[] = []) {
  const [state, setState] = useState<AsyncState<T>>({ data: null, loading: true, error: null })
  const requestId = useRef(0)

  const reload = useCallback(async () => {
    const current = ++requestId.current
    setState((previous) => ({ ...previous, loading: true, error: null }))
    try {
      const data = await loader()
      if (current === requestId.current) setState({ data, loading: false, error: null })
      return data
    } catch (error) {
      const normalized = error instanceof Error ? error : new Error('加载失败')
      if (current === requestId.current) setState({ data: null, loading: false, error: normalized })
      return undefined
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies)

  useEffect(() => {
    void reload()
    return () => { requestId.current += 1 }
  }, [reload])

  return { ...state, reload }
}
