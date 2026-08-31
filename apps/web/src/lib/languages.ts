import type { EditorLanguage, KernelLanguage, ProblemImplementation } from '../domain/types'

export interface LanguageMetadata {
  label: string
  shortLabel: string
  editorLanguage: EditorLanguage
  fileName: string
  fileExtension: ProblemImplementation['file_extension']
  diagnosticsLabel: string
  runtime: 'cuda' | 'triton' | 'torch'
}

export const implementationLanguages = [
  'cuda_cpp',
  'triton_python',
  'torch_python',
] as const satisfies readonly KernelLanguage[]

export const languageMetadata = {
  cuda_cpp: {
    label: 'CUDA C++',
    shortLabel: 'CUDA C++',
    editorLanguage: 'cpp',
    fileName: 'solution.cu',
    fileExtension: '.cu',
    diagnosticsLabel: 'NVCC 诊断',
    runtime: 'cuda',
  },
  triton_python: {
    label: 'Triton (Python)',
    shortLabel: 'Triton',
    editorLanguage: 'python',
    fileName: 'solution.py',
    fileExtension: '.py',
    diagnosticsLabel: 'Triton / Python 诊断',
    runtime: 'triton',
  },
  torch_python: {
    label: 'PyTorch (Python)',
    shortLabel: 'PyTorch',
    editorLanguage: 'python',
    fileName: 'solution.py',
    fileExtension: '.py',
    diagnosticsLabel: 'Python / PyTorch 诊断',
    runtime: 'torch',
  },
} as const satisfies Record<KernelLanguage, LanguageMetadata>

export function isKernelLanguage(value: unknown): value is KernelLanguage {
  return typeof value === 'string' && (implementationLanguages as readonly string[]).includes(value)
}

export function languageLabel(language: KernelLanguage): string {
  return languageMetadata[language].label
}
