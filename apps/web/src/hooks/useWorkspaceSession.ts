import { useCallback, useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { KernelLanguage, ProblemDetail } from '../domain/types'
import { implementationLanguages, isKernelLanguage } from '../lib/languages'
import { useWorkspaceDraft } from './useWorkspaceDraft'

export function useWorkspaceSession(problemId: string, problem: ProblemDetail | null) {
  const [searchParams, setSearchParams] = useSearchParams()
  // Async resources may retain their previous value while the next slug loads.
  // Never initialize a new draft using the previous problem's implementation.
  const currentProblem = problem?.slug === problemId ? problem : null
  const supportedLanguages = useMemo(() => (
    implementationLanguages.filter((language) => currentProblem?.implementations[language])
  ), [currentProblem])
  const requested = searchParams.get('language')
  const language = isKernelLanguage(requested) && supportedLanguages.includes(requested)
    ? requested
    : currentProblem?.default_language ?? supportedLanguages[0] ?? 'cuda_cpp'
  const implementation = currentProblem?.implementations[language]
  const draft = useWorkspaceDraft({ problemId, language, starterCode: implementation?.starter_code })

  useEffect(() => {
    if (!currentProblem || searchParams.get('language') === language) return
    const next = new URLSearchParams(searchParams)
    next.set('language', language)
    setSearchParams(next, { replace: true })
  }, [currentProblem, language, searchParams, setSearchParams])

  const selectLanguage = useCallback((nextLanguage: KernelLanguage) => {
    if (nextLanguage === language || !supportedLanguages.includes(nextLanguage)) return false
    draft.flush()
    const next = new URLSearchParams(searchParams)
    next.set('language', nextLanguage)
    setSearchParams(next)
    return true
  }, [draft.flush, language, searchParams, setSearchParams, supportedLanguages])

  return { language, implementation, supportedLanguages, selectLanguage, draft }
}
