import Editor, { DiffEditor, type BeforeMount } from '@monaco-editor/react'
import { memo } from 'react'
import type { KernelLanguage } from '../domain/types'
import { useTheme } from '../hooks/useTheme'
import { languageMetadata } from '../lib/languages'
import '../monaco'

const configure: BeforeMount = (monaco) => {
  monaco.editor.defineTheme('myleetgpu-light', {
    base: 'vs',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '69717C' },
      { token: 'keyword', foreground: '486B80' },
      { token: 'number', foreground: '80674F' },
      { token: 'type', foreground: '576E64' },
      { token: 'string', foreground: '6D7151' },
      { token: 'delimiter', foreground: '626A75' },
    ],
    colors: {
      'editor.background': '#FFFFFF',
      'editor.foreground': '#262B33',
      'editor.lineHighlightBackground': '#F6F7F9',
      'editor.selectionBackground': '#DCE6ED',
      'editor.inactiveSelectionBackground': '#E9EEF2',
      'editorCursor.foreground': '#486B80',
      'editorLineNumber.foreground': '#929AA4',
      'editorLineNumber.activeForeground': '#486B80',
      'editorIndentGuide.background1': '#EDF0F3',
      'editorIndentGuide.activeBackground1': '#CCD5DC',
      'editorWidget.background': '#FFFFFF',
      'editorWidget.border': '#DDE2E7',
      'editorGutter.background': '#FFFFFF',
      'editorBracketHighlight.foreground1': '#626A75',
      'editorBracketHighlight.foreground2': '#626A75',
      'editorBracketHighlight.foreground3': '#626A75',
      'editorBracketHighlight.foreground4': '#626A75',
      'editorBracketHighlight.foreground5': '#626A75',
      'editorBracketHighlight.foreground6': '#626A75',
      'editorBracketMatch.background': '#486B8010',
      'editorBracketMatch.border': '#AEBCC7',
      'diffEditor.insertedTextBackground': '#71917C2B',
      'diffEditor.insertedLineBackground': '#7D96821A',
      'diffEditor.removedTextBackground': '#AF77752B',
      'diffEditor.removedLineBackground': '#AD817B1A',
      'diffEditorGutter.insertedLineBackground': '#7D96821A',
      'diffEditorGutter.removedLineBackground': '#AD817B1A',
      'diffEditorOverview.insertedForeground': '#93AA99',
      'diffEditorOverview.removedForeground': '#C19A94',
    },
  })
  monaco.editor.defineTheme('myleetgpu-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '919AA5' },
      { token: 'keyword', foreground: '9BB2C4' },
      { token: 'number', foreground: 'C0AA91' },
      { token: 'type', foreground: 'A1B7AA' },
      { token: 'string', foreground: 'B0B59A' },
      { token: 'delimiter', foreground: 'AEB6BF' },
    ],
    colors: {
      'editor.background': '#202327',
      'editor.foreground': '#D5D9DF',
      'editor.lineHighlightBackground': '#272B30',
      'editor.selectionBackground': '#3C4C59',
      'editor.inactiveSelectionBackground': '#343C44',
      'editorCursor.foreground': '#9BB2C4',
      'editorLineNumber.foreground': '#747E89',
      'editorLineNumber.activeForeground': '#BCC7D1',
      'editorIndentGuide.background1': '#2D3339',
      'editorIndentGuide.activeBackground1': '#49525B',
      'editorWidget.background': '#272B30',
      'editorWidget.border': '#3D454E',
      'editorGutter.background': '#202327',
      'editorBracketHighlight.foreground1': '#AEB6BF',
      'editorBracketHighlight.foreground2': '#AEB6BF',
      'editorBracketHighlight.foreground3': '#AEB6BF',
      'editorBracketHighlight.foreground4': '#AEB6BF',
      'editorBracketHighlight.foreground5': '#AEB6BF',
      'editorBracketHighlight.foreground6': '#AEB6BF',
      'editorBracketMatch.background': '#9BB2C410',
      'editorBracketMatch.border': '#516170',
      'diffEditor.insertedTextBackground': '#73998230',
      'diffEditor.insertedLineBackground': '#64836D20',
      'diffEditor.removedTextBackground': '#AD817B30',
      'diffEditor.removedLineBackground': '#916E6B20',
      'diffEditorGutter.insertedLineBackground': '#64836D20',
      'diffEditorGutter.removedLineBackground': '#916E6B20',
      'diffEditorOverview.insertedForeground': '#647D6E',
      'diffEditorOverview.removedForeground': '#916E6B',
    },
  })
}

const isTest = typeof navigator !== 'undefined' && navigator.userAgent.toLowerCase().includes('jsdom')

export const CodeEditor = memo(function CodeEditor({ value, language, problemId, onChange, readOnly = false }: {
  value: string
  language: KernelLanguage
  problemId: string
  onChange?: (value: string) => void
  readOnly?: boolean
}) {
  const { theme } = useTheme()
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
      theme={`myleetgpu-${theme}`}
      value={value}
      onChange={(next) => onChange?.(next ?? '')}
      options={{
        readOnly,
        ariaLabel: `${metadata.label} 代码编辑器`,
        automaticLayout: true,
        fontFamily: "'SFMono-Regular', Menlo, Monaco, Consolas, 'Liberation Mono', monospace",
        fontSize: 14,
        lineHeight: 24,
        fontLigatures: false,
        minimap: { enabled: false },
        padding: { top: 14, bottom: 16 },
        renderLineHighlight: 'line',
        smoothScrolling: false,
        scrollBeyondLastLine: false,
        wordWrap: 'off',
        tabSize: 2,
        insertSpaces: true,
        bracketPairColorization: { enabled: false },
        renderWhitespace: 'selection',
        overviewRulerBorder: false,
        stickyScroll: { enabled: false },
      }}
    />
  )
})

export function CodeDiff({ original, modified, language, originalLabel, modifiedLabel }: {
  original: string
  modified: string
  language: KernelLanguage
  originalLabel?: string
  modifiedLabel?: string
}) {
  const { theme } = useTheme()
  const metadata = languageMetadata[language]
  if (isTest) {
    return <div className="test-diff" data-editor-language={metadata.editorLanguage}><pre aria-label={originalLabel}>{original}</pre><pre aria-label={modifiedLabel}>{modified}</pre></div>
  }
  return (
    <DiffEditor
      beforeMount={configure}
      height="100%"
      language={metadata.editorLanguage}
      theme={`myleetgpu-${theme}`}
      original={original}
      modified={modified}
      options={{
        automaticLayout: true,
        readOnly: true,
        renderSideBySide: true,
        minimap: { enabled: false },
        fontFamily: "'SFMono-Regular', Menlo, Monaco, Consolas, 'Liberation Mono', monospace",
        fontSize: 14,
        lineHeight: 24,
        scrollBeyondLastLine: false,
        originalEditable: false,
        bracketPairColorization: { enabled: false },
        stickyScroll: { enabled: false },
      }}
    />
  )
}
