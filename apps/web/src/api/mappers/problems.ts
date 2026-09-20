import type { KernelLanguage, ProblemDetail, ProblemImplementation, ProblemSummary } from '../../domain/types'
import { implementationLanguages, isKernelLanguage, languageMetadata } from '../../lib/languages'
import { asKernelLanguage, asRecord, asString } from './common'

function normalizeSignature(value: unknown): string | undefined {
  if (typeof value === 'string') return value || undefined
  const raw = asRecord(value)
  return asString(raw.declaration ?? raw.symbol) || undefined
}

function normalizeImplementation(value: unknown, language: KernelLanguage): ProblemImplementation {
  const raw = asRecord(value)
  const metadata = languageMetadata[language]
  const fileExtension = asString(raw.file_extension ?? raw.source_suffix, metadata.fileExtension)
  const editorLanguage = asString(raw.editor_language, metadata.editorLanguage)
  return {
    language,
    display_name: asString(raw.display_name, metadata.label),
    file_extension: fileExtension === '.py' || fileExtension === '.cu' || fileExtension === '.cpp' ? fileExtension : metadata.fileExtension,
    editor_language: editorLanguage === 'python' || editorLanguage === 'cpp' ? editorLanguage : metadata.editorLanguage,
    starter_code: asString(raw.starter_code),
    signature: normalizeSignature(raw.signature),
    instructions_markdown: asString(raw.instructions_markdown ?? raw.statement_appendix ?? raw.instructions) || undefined,
  }
}

export function normalizeProblemSummary(value: unknown): ProblemSummary {
  const raw = asRecord(value)
  const languages = Array.isArray(raw.languages)
    ? raw.languages.filter(isKernelLanguage).filter((item, index, all) => all.indexOf(item) === index)
    : undefined
  return {
    slug: asString(raw.slug ?? raw.id),
    title: asString(raw.title, '未命名题目'),
    difficulty: asString(raw.difficulty, 'easy'),
    revision: asString(raw.revision, '1'),
    summary: asString(raw.summary),
    languages,
  }
}

function displayConstraint(name: string, value: unknown): string {
  const detail = asRecord(value)
  if ('min' in detail || 'max' in detail) {
    const bounds = [detail.min !== undefined ? `min=${detail.min}` : '', detail.max !== undefined ? `max=${detail.max}` : ''].filter(Boolean)
    return `${name}: ${bounds.join('，')}`
  }
  if (Object.keys(detail).length) {
    return `${name}: ${Object.entries(detail).map(([key, item]) => `${key}=${String(item)}`).join('，')}`
  }
  return `${name}: ${asString(value)}`
}

export function normalizeProblemDetail(value: unknown): ProblemDetail {
  const raw = asRecord(value)
  const implementationRaw = asRecord(raw.implementations)
  const implementations: Partial<Record<KernelLanguage, ProblemImplementation>> = {}
  for (const language of implementationLanguages) {
    if (implementationRaw[language] !== undefined) {
      implementations[language] = normalizeImplementation(implementationRaw[language], language)
    }
  }
  const legacyLanguage = asKernelLanguage(raw.language)
  if (!Object.keys(implementations).length) {
    implementations[legacyLanguage] = normalizeImplementation({
      language: legacyLanguage,
      starter_code: raw.starter_code,
      signature: raw.signature,
      instructions_markdown: raw.instructions_markdown ?? raw.statement_appendix,
    }, legacyLanguage)
  }
  const requestedDefault = asKernelLanguage(raw.default_language ?? raw.language)
  const defaultLanguage = implementations[requestedDefault]
    ? requestedDefault
    : implementationLanguages.find((language) => implementations[language]) ?? 'cuda_cpp'
  const defaultImplementation = implementations[defaultLanguage] ?? normalizeImplementation({}, defaultLanguage)
  const constraints = Array.isArray(raw.constraints)
    ? raw.constraints.map((item) => asString(item))
    : Object.entries(asRecord(raw.constraints)).map(([name, item]) => displayConstraint(name, item))
  const benchmarkRaw = asRecord(raw.benchmark)
  const sizesRaw = Array.isArray(benchmarkRaw.input_sizes)
    ? benchmarkRaw.input_sizes
    : Array.isArray(benchmarkRaw.sizes)
      ? benchmarkRaw.sizes
      : []
  const inputSizes = sizesRaw.map((item) => {
    const itemRecord = asRecord(item)
    return itemRecord.label !== undefined ? asString(itemRecord.label) : typeof item === 'number' ? item : asString(item)
  })
  return {
    ...normalizeProblemSummary(raw),
    languages: implementationLanguages.filter((language) => implementations[language]),
    statement_markdown: asString(raw.statement_markdown),
    default_language: defaultLanguage,
    implementations,
    language: defaultLanguage,
    starter_code: defaultImplementation.starter_code,
    signature: defaultImplementation.signature,
    constraints,
    benchmark: { ...benchmarkRaw, input_sizes: inputSizes },
  }
}
