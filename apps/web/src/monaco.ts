import { loader } from '@monaco-editor/react'
import * as monaco from 'monaco-editor/esm/vs/editor/editor.api'
import 'monaco-editor/esm/vs/basic-languages/cpp/cpp.contribution'
import 'monaco-editor/esm/vs/basic-languages/python/python.contribution'
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker'

type MonacoScope = typeof globalThis & {
  MonacoEnvironment?: {
    getWorker: (_moduleId: string, _label: string) => Worker
  }
}

;(globalThis as MonacoScope).MonacoEnvironment = {
  getWorker: () => new EditorWorker(),
}

// Supplying the installed ESM build prevents @monaco-editor/react from loading
// editor code from a public CDN. MyLeetGpu therefore remains usable offline.
loader.config({ monaco })
