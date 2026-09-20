import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { KernelLanguage } from '../domain/types'
import { readLocalDraft, saveLocalDraft } from '../lib/drafts'

type DraftStatus = 'idle' | 'saving' | 'saved' | 'local-only'

interface DraftSession {
  problemId: string
  language: KernelLanguage
  source: string
  revision: number
  saveSequence: number
  active: boolean
  ready: boolean
}

interface DraftState {
  session?: DraftSession
  source: string
  ready: boolean
  savedAt?: string
  status: DraftStatus
}

interface WorkspaceDraftOptions {
  problemId: string
  language: KernelLanguage
  // Undefined means the requested problem/implementation has not loaded yet.
  starterCode?: string
}

const emptyState: DraftState = { source: '', ready: false, status: 'idle' }

export function useWorkspaceDraft({ problemId, language, starterCode }: WorkspaceDraftOptions) {
  const [state, setState] = useState<DraftState>(emptyState)
  const currentSession = useRef<DraftSession>()

  useEffect(() => {
    if (starterCode === undefined) return
    const local = readLocalDraft(problemId, language)
    const session: DraftSession = {
      problemId, language, source: local?.source ?? starterCode,
      revision: 0, saveSequence: 0, active: true, ready: false,
    }
    currentSession.current = session
    setState({ session, source: session.source, ready: false, savedAt: local?.updatedAt, status: 'idle' })

    // A session owns both its key and its mutable source. Its cleanup must never
    // read another route's source from a shared, subsequently reassigned ref.
    const flushLocal = () => {
      if (session.ready || session.revision > 0) saveLocalDraft(session.problemId, session.language, session.source)
    }
    window.addEventListener('pagehide', flushLocal)
    void api.drafts.get(problemId, language).then((remote) => {
      if (!session.active) return
      const localTime = Date.parse(local?.updatedAt ?? '') || 0
      const remoteTime = Date.parse(remote.updated_at) || 0
      if (session.revision === 0 && remoteTime > localTime) {
        session.source = remote.source
        saveLocalDraft(problemId, language, remote.source)
        setState((previous) => ({ ...previous, source: remote.source, savedAt: remote.updated_at }))
      }
      setState((previous) => ({ ...previous, status: 'saved' }))
    }).catch(() => {
      if (session.active) setState((previous) => ({ ...previous, status: 'local-only' }))
    }).finally(() => {
      if (!session.active) return
      session.ready = true
      setState((previous) => ({ ...previous, ready: true }))
    })

    return () => {
      window.removeEventListener('pagehide', flushLocal)
      flushLocal()
      session.active = false
      session.saveSequence += 1
      if (currentSession.current === session) currentSession.current = undefined
    }
  }, [problemId, language, starterCode])

  useEffect(() => {
    const session = state.session
    if (!session?.active || !state.ready) return
    const sequence = ++session.saveSequence
    const isCurrent = () => session.active && sequence === session.saveSequence
    const localTimer = window.setTimeout(() => {
      if (!isCurrent()) return
      const saved = saveLocalDraft(session.problemId, session.language, state.source)
      setState((previous) => ({ ...previous, savedAt: saved.updatedAt, status: 'saving' }))
    }, 350)
    const remoteTimer = window.setTimeout(() => {
      if (!isCurrent()) return
      void api.drafts.save(session.problemId, session.language, state.source).then((draft) => {
        if (isCurrent()) setState((previous) => ({ ...previous, savedAt: draft.updated_at, status: 'saved' }))
      }).catch(() => {
        if (isCurrent()) setState((previous) => ({ ...previous, status: 'local-only' }))
      })
    }, 1_100)
    return () => {
      window.clearTimeout(localTimer)
      window.clearTimeout(remoteTimer)
      session.saveSequence += 1
    }
  }, [state.session, state.ready, state.source])

  const updateSource = useCallback((source: string) => {
    const session = currentSession.current
    if (!session?.active) return
    session.revision += 1
    session.saveSequence += 1
    session.source = source
    setState((previous) => ({ ...previous, source }))
  }, [])

  const flush = useCallback(() => {
    const session = currentSession.current
    if (!session?.active || !session.ready) return
    session.saveSequence += 1
    const saved = saveLocalDraft(session.problemId, session.language, session.source)
    setState((previous) => ({ ...previous, savedAt: saved.updatedAt }))
    void api.drafts.save(session.problemId, session.language, session.source).catch(() => undefined)
  }, [])

  const getSource = useCallback(() => currentSession.current?.source ?? '', [])
  const matchesRoute = starterCode !== undefined && state.session?.active
    && state.session.problemId === problemId && state.session.language === language

  return {
    source: matchesRoute ? state.source : '',
    ready: Boolean(matchesRoute && state.ready),
    savedAt: matchesRoute ? state.savedAt : undefined,
    status: matchesRoute ? state.status : 'idle' as DraftStatus,
    updateSource,
    getSource,
    flush,
  }
}
