import type { KernelLanguage } from '../../domain/types'
import { isKernelLanguage } from '../../lib/languages'

export function listItems<T>(payload: { items?: T[] } | T[]): T[] {
  return Array.isArray(payload) ? payload : payload.items ?? []
}

type JsonRecord = Record<string, unknown>

export function asRecord(value: unknown): JsonRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as JsonRecord : {}
}

export function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : value === null || value === undefined ? fallback : String(value)
}

export function asNumber(value: unknown, fallback = 0): number {
  const parsed = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

export function asDate(value: unknown): string {
  const text = asString(value)
  if (!text) return ''
  return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?$/.test(text) ? `${text}Z` : text
}

export function asKernelLanguage(value: unknown, fallback: KernelLanguage = 'cuda_cpp'): KernelLanguage {
  return isKernelLanguage(value) ? value : fallback
}
