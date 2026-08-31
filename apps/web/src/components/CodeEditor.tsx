import Editor, { DiffEditor, type BeforeMount } from '@monaco-editor/react'
import type { KernelLanguage } from '../domain/types'
import { languageMetadata } from '../lib/languages'
import '../monaco'

const configure: BeforeMount = (monaco) => {
  monaco.editor.defineTheme('myleetgpu-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '687487', fontStyle: 'italic' },
      { token: 'keyword', foreground: 'C89BFF' },
      { token: 'number', foreground: '7DD3FC' },
      { token: 'type', foreground: '62D6A8' },
    ],
    colors: {
      'editor.background': '#0D1118',
      'editor.foreground': '#D8DEE9',
      'editor.lineHighlightBackground': '#131A24',
      'editor.selectionBackground': '#5B49D955',
      'editorLineNumber.foreground': '#465163',
      'editorLineNumber.activeForeground': '#97A2B5',
      'editorIndentGuide.background1': '#202937',
      'editorIndentGuide.activeBackground1': '#354154',
    },
  })
}

const isTest = typeof navigator !== 'undefined' && navigator.userAgent.toLowerCase().includes('jsdom')

export function CodeEditor({ value, language, problemId, onChange, readOnly = false }: {
  value: string
  language: KernelLanguage
  problemId: string
  onChange?: (value: string) => void
  readOnly?: boolean
}) {
  const metadata = languageMetadata[language]
  const modelPath = `/problems/${encodeURIComponent(problemId)}/${language}/${metadata.fileName}`
  if (isTest) {
    return <textarea aria-label={`${metadata.label} 代码编辑器`} data-editor-language={metadata.editorLanguage} data-editor-path={modelPath} className="test-code-editor" value={value} readOnly={readOnly} onChange={(event) => onChange?.(event.target.value)} />
  }
  return (
    <Editor
      beforeMount={configure}
      height="100%"
      language={metadata.editorLanguage}
      path={modelPath}
      theme="myleetgpu-dark"
      value={value}
      onChange={(next) => onChange?.(next ?? '')}
      options={{
        readOnly,
        automaticLayout: true,
        fontFamily: "'JetBrains Mono', 'Cascadia Code', Consolas, monospace",
        fontSize: 13.5,
        lineHeight: 22,
        minimap: { enabled: false },
        padding: { top: 14, bottom: 16 },
        renderLineHighlight: 'line',
        smoothScrolling: true,
        scrollBeyondLastLine: false,
        wordWrap: 'off',
        tabSize: 2,
        insertSpaces: true,
        bracketPairColorization: { enabled: true },
      }}
    />
  )
}

export function CodeDiff({ original, modified, language, originalLabel, modifiedLabel }: {
  original: string
  modified: string
  language: KernelLanguage
  originalLabel?: string
  modifiedLabel?: string
}) {
  const metadata = languageMetadata[language]
  if (isTest) {
    return <div className="test-diff" data-editor-language={metadata.editorLanguage}><pre aria-label={originalLabel}>{original}</pre><pre aria-label={modifiedLabel}>{modified}</pre></div>
  }
  return (
    <DiffEditor
      beforeMount={configure}
      height="100%"
      language={metadata.editorLanguage}
      theme="myleetgpu-dark"
      original={original}
      modified={modified}
      options={{
        automaticLayout: true,
        readOnly: true,
        renderSideBySide: true,
        minimap: { enabled: false },
        fontFamily: "'JetBrains Mono', 'Cascadia Code', Consolas, monospace",
        fontSize: 12.5,
        lineHeight: 20,
        scrollBeyondLastLine: false,
        originalEditable: false,
      }}
    />
  )
}
