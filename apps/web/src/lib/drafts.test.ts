import { beforeEach, describe, expect, it } from 'vitest'
import { readLocalDraft, saveLocalDraft } from './drafts'

describe('local drafts', () => {
  beforeEach(() => localStorage.clear())

  it('persists independent drafts per problem and implementation language', () => {
    const cuda = saveLocalDraft('vector-add', 'cuda_cpp', '__global__ void solve() {}')
    const triton = saveLocalDraft('vector-add', 'triton_python', '@triton.jit\ndef solve(): ...')
    expect(readLocalDraft('vector-add', 'cuda_cpp')).toEqual(cuda)
    expect(readLocalDraft('vector-add', 'triton_python')).toEqual(triton)
    expect(readLocalDraft('reduction', 'cuda_cpp')).toBeNull()
  })

  it('ignores damaged local storage records', () => {
    localStorage.setItem('myleetgpu:draft:vector-add:triton_python', '{broken')
    expect(readLocalDraft('vector-add', 'triton_python')).toBeNull()
  })

  it('migrates a legacy problem draft only into CUDA C++ storage', () => {
    localStorage.setItem('myleetgpu:draft:vector-add', JSON.stringify({ source: '// legacy', updatedAt: '2026-01-01T00:00:00Z' }))
    expect(readLocalDraft('vector-add', 'triton_python')).toBeNull()
    expect(readLocalDraft('vector-add', 'cuda_cpp')).toEqual({ language: 'cuda_cpp', source: '// legacy', updatedAt: '2026-01-01T00:00:00Z' })
    expect(localStorage.getItem('myleetgpu:draft:vector-add')).toBeNull()
  })
})
