import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type CreateJobInput } from '../api/client'
import type { Job } from '../domain/types'

const TERMINAL = new Set(['succeeded', 'failed', 'timed_out', 'cancelled', 'system_error'])

export function useJob(onSettled?: (job: Job) => void) {
  const [job, setJob] = useState<Job | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const deadlineRef = useRef(0)
  const onSettledRef = useRef(onSettled)
  onSettledRef.current = onSettled

  const start = useCallback(async (input: CreateJobInput) => {
    setSubmitting(true)
    setError(null)
    try {
      const created = await api.jobs.create(input)
      deadlineRef.current = Date.now() + 10 * 60_000
      setJob(created)
      return created
    } catch (caught) {
      const normalized = caught instanceof Error ? caught : new Error('提交任务失败')
      setError(normalized)
      throw normalized
    } finally {
      setSubmitting(false)
    }
  }, [])

  useEffect(() => {
    if (!job || TERMINAL.has(job.status)) return
    let cancelled = false
    let timer: number | undefined

    const poll = async () => {
      try {
        if (Date.now() > deadlineRef.current) {
          const timedOut: Job = {
            ...job,
            status: 'timed_out',
            error: { message: '等待任务状态超过 10 分钟。后台任务可能仍在运行，请稍后刷新。' },
          }
          if (!cancelled) {
            setJob(timedOut)
            onSettledRef.current?.(timedOut)
          }
          return
        }
        const next = await api.jobs.get(job.id)
        if (cancelled) return
        setJob(next)
        if (TERMINAL.has(next.status)) {
          onSettledRef.current?.(next)
        } else {
          timer = window.setTimeout(poll, 800)
        }
      } catch (caught) {
        if (cancelled) return
        const normalized = caught instanceof Error ? caught : new Error('读取任务状态失败')
        setError(normalized)
        timer = window.setTimeout(poll, 2_000)
      }
    }

    timer = window.setTimeout(poll, 500)
    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
    }
  }, [job?.id, job?.status])

  const clear = useCallback(() => {
    setJob(null)
    setError(null)
  }, [])

  return { job, submitting, error, start, clear, busy: submitting || (!!job && !TERMINAL.has(job.status)) }
}
