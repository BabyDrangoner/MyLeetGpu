export async function sourceHash(source: string): Promise<string> {
  const normalized = source.replace(/\r\n?/g, '\n')
  if (globalThis.crypto?.subtle) {
    const encoded = new TextEncoder().encode(normalized)
    const digest = await globalThis.crypto.subtle.digest('SHA-256', encoded)
    return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
  }
  // Deterministic test/browser fallback; the server remains authoritative.
  let hash = 2166136261
  for (let index = 0; index < normalized.length; index += 1) {
    hash ^= normalized.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return `local-${(hash >>> 0).toString(16).padStart(8, '0')}`
}
