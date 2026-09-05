import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'

export const WORKSPACE_LAYOUT_KEY = 'myleetgpu:workspace-layout:v1'
const DEFAULT_LAYOUT = { statement: 36, editor: 73 }
const HANDLE_SIZE = 12
type Layout = typeof DEFAULT_LAYOUT
type Panel = keyof Layout

function readLayout(): Layout {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(WORKSPACE_LAYOUT_KEY) ?? 'null')
    if (!saved || typeof saved !== 'object') return DEFAULT_LAYOUT
    const { statement, editor } = saved as Partial<Layout>
    const valid = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value > 0 && value < 100
    return {
      statement: valid(statement) ? statement : DEFAULT_LAYOUT.statement,
      editor: valid(editor) ? editor : DEFAULT_LAYOUT.editor,
    }
  } catch {
    return DEFAULT_LAYOUT
  }
}

function useContainerSize(axis: 'width' | 'height') {
  const ref = useRef<HTMLElement | null>(null)
  const [element, setElement] = useState<HTMLElement | null>(null)
  const [size, setSize] = useState(0)
  const attach = useCallback((node: HTMLElement | null) => {
    ref.current = node
    setElement(node)
  }, [])

  useLayoutEffect(() => {
    if (!element) return
    const measure = () => setSize(element.getBoundingClientRect()[axis])
    measure()
    const observer = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(measure)
    observer?.observe(element)
    window.addEventListener('resize', measure)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [axis, element])

  return { ref, attach, size }
}

export function splitBounds(size: number, firstMinimum: number, secondMinimum: number) {
  const available = size - HANDLE_SIZE
  if (available <= 0) return { min: 0, max: 100 }
  // When the viewport is very short, share the available space without overflowing.
  const scale = Math.min(1, available / (firstMinimum + secondMinimum))
  const min = firstMinimum * scale / available * 100
  return { min, max: Math.max(min, 100 - secondMinimum * scale / available * 100) }
}

export function useWorkspaceLayout() {
  const [layout, setLayout] = useState(readLayout)
  const latestLayout = useRef(layout)
  const workspace = useContainerSize('width')
  const column = useContainerSize('height')

  useEffect(() => {
    // Keep storage work out of the pointer-move hot path.
    latestLayout.current = layout
    const persist = () => {
      try { localStorage.setItem(WORKSPACE_LAYOUT_KEY, JSON.stringify(layout)) } catch { /* Layout still works without storage. */ }
    }
    const timer = window.setTimeout(persist, 150)
    return () => window.clearTimeout(timer)
  }, [layout])

  useEffect(() => {
    const flush = () => {
      try { localStorage.setItem(WORKSPACE_LAYOUT_KEY, JSON.stringify(latestLayout.current)) } catch { /* Storage may be unavailable. */ }
    }
    window.addEventListener('pagehide', flush)
    return () => {
      window.removeEventListener('pagehide', flush)
      flush()
    }
  }, [])

  const update = (panel: Panel, value: number) => {
    if (!Number.isFinite(value)) return
    setLayout((current) => ({ ...current, [panel]: Math.min(99.9, Math.max(0.1, value)) }))
  }
  const split = (panel: Panel, bounds: { min: number; max: number }) => ({
    ...bounds,
    value: Math.min(bounds.max, Math.max(bounds.min, layout[panel])),
    onChange: (value: number) => update(panel, value),
    onReset: () => update(panel, DEFAULT_LAYOUT[panel]),
  })
  // Clamp only the rendered split, preserving preferences across focus mode and viewport changes.
  const statement = split('statement', splitBounds(workspace.size, 280, 440))
  const editor = split('editor', splitBounds(column.size, 260, 140))

  return {
    workspace,
    column,
    statement,
    editor,
    reset: () => setLayout({ ...DEFAULT_LAYOUT }),
    workspaceStyle: { '--statement-share': `${statement.value}fr`, '--code-share': `${100 - statement.value}fr` } as CSSProperties,
    editorStyle: { '--editor-share': `${editor.value}fr`, '--output-share': `${100 - editor.value}fr` } as CSSProperties,
  }
}
