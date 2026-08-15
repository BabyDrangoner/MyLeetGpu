import { beforeEach, describe, expect, it } from 'vitest'
import { readLocalDraft, saveLocalDraft } from './drafts'

describe('local drafts', () => {
  beforeEach(() => localStorage.clear())

  it('persists a draft per problem', () => {
    const saved = saveLocalDraft('vector-add', '__global__ void solve() {}')
    expect(readLocalDraft('vector-add')).toEqual(saved)
    expect(readLocalDraft('reduction')).toBeNull()
  })

  it('ignores damaged local storage records', () => {
    localStorage.setItem('myleetgpu:draft:vector-add', '{broken')
    expect(readLocalDraft('vector-add')).toBeNull()
  })
})
