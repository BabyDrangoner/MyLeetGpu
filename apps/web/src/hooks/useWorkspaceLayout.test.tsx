import { act, cleanup, render, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { splitBounds, useWorkspaceLayout, WORKSPACE_LAYOUT_KEY } from './useWorkspaceLayout'

beforeEach(() => {
  localStorage.clear()
  vi.useFakeTimers()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('workspace layout preferences', () => {
  it('starts with the default proportions and matching CSS tracks', () => {
    const { result } = renderHook(useWorkspaceLayout)
    expect(result.current.statement.value).toBe(36)
    expect(result.current.editor.value).toBe(73)
    expect(result.current.workspaceStyle).toEqual({ '--statement-share': '36fr', '--code-share': '64fr' })
    expect(result.current.editorStyle).toEqual({ '--editor-share': '73fr', '--output-share': '27fr' })
  })

  it.each([
    ['{"statement":42,"editor":68}', 42, 68],
    ['{"statement":48,"editor":"bad"}', 48, 73],
    ['{"statement":null,"editor":62}', 36, 62],
    ['{"statement":0,"editor":100}', 36, 73],
    ['{"statement":1e400,"editor":-1}', 36, 73],
    ['{broken', 36, 73],
    ['null', 36, 73],
    ['42', 36, 73],
  ])('validates saved layout %s field by field', (saved, statement, editor) => {
    localStorage.setItem(WORKSPACE_LAYOUT_KEY, saved)
    const { result } = renderHook(useWorkspaceLayout)
    expect(result.current.statement.value).toBe(statement)
    expect(result.current.editor.value).toBe(editor)
  })

  it('debounces changes, persists the latest pair, and restores them on remount', () => {
    const write = vi.spyOn(Storage.prototype, 'setItem')
    const { result, unmount } = renderHook(useWorkspaceLayout)
    act(() => result.current.statement.onChange(44))
    act(() => vi.advanceTimersByTime(100))
    act(() => result.current.editor.onChange(65))
    act(() => vi.advanceTimersByTime(149))
    expect(write).not.toHaveBeenCalled()
    act(() => vi.advanceTimersByTime(1))
    expect(write).toHaveBeenCalledTimes(1)
    expect(JSON.parse(localStorage.getItem(WORKSPACE_LAYOUT_KEY)!)).toEqual({ statement: 44, editor: 65 })

    unmount()
    const restored = renderHook(useWorkspaceLayout)
    expect(restored.result.current.statement.value).toBe(44)
    expect(restored.result.current.editor.value).toBe(65)
  })

  it('flushes the latest unsaved values on pagehide and unmount', () => {
    const write = vi.spyOn(Storage.prototype, 'setItem')
    const { result, unmount } = renderHook(useWorkspaceLayout)
    act(() => result.current.statement.onChange(41))
    act(() => window.dispatchEvent(new Event('pagehide')))
    expect(JSON.parse(localStorage.getItem(WORKSPACE_LAYOUT_KEY)!)).toEqual({ statement: 41, editor: 73 })

    act(() => result.current.editor.onChange(64))
    unmount()
    expect(JSON.parse(localStorage.getItem(WORKSPACE_LAYOUT_KEY)!)).toEqual({ statement: 41, editor: 64 })
    write.mockClear()
    act(() => vi.advanceTimersByTime(150))
    expect(write).not.toHaveBeenCalled()
  })

  it('resets a single split independently and resets the entire layout', () => {
    localStorage.setItem(WORKSPACE_LAYOUT_KEY, '{"statement":45,"editor":60}')
    const { result } = renderHook(useWorkspaceLayout)
    act(() => result.current.statement.onReset())
    expect(result.current.statement.value).toBe(36)
    expect(result.current.editor.value).toBe(60)
    act(() => result.current.statement.onChange(40))
    act(() => result.current.editor.onReset())
    expect(result.current.statement.value).toBe(40)
    expect(result.current.editor.value).toBe(73)
    act(() => result.current.editor.onChange(67))
    act(() => result.current.reset())
    expect(result.current.statement.value).toBe(36)
    expect(result.current.editor.value).toBe(73)
    act(() => vi.advanceTimersByTime(150))
    expect(JSON.parse(localStorage.getItem(WORKSPACE_LAYOUT_KEY)!)).toEqual({ statement: 36, editor: 73 })
  })

  it('ignores non-finite updates and bounds finite values before persisting', () => {
    const { result } = renderHook(useWorkspaceLayout)
    act(() => result.current.statement.onChange(Number.NaN))
    act(() => result.current.editor.onChange(Infinity))
    expect(result.current.statement.value).toBe(36)
    expect(result.current.editor.value).toBe(73)
    act(() => result.current.statement.onChange(-10))
    act(() => result.current.editor.onChange(110))
    expect(result.current.statement.value).toBe(0.1)
    expect(result.current.editor.value).toBe(99.9)
  })

  it('remains usable when browser storage is denied', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('Storage denied') })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('Storage denied') })
    const { result, unmount } = renderHook(useWorkspaceLayout)
    expect(result.current.statement.value).toBe(36)
    act(() => result.current.statement.onChange(43))
    expect(result.current.statement.value).toBe(43)
    expect(() => act(() => vi.advanceTimersByTime(150))).not.toThrow()
    expect(() => act(() => window.dispatchEvent(new Event('pagehide')))).not.toThrow()
    expect(unmount).not.toThrow()
  })
})

describe('workspace split measurements', () => {
  it('reserves the handle and enforces both minimum panel sizes', () => {
    const bounds = splitBounds(1012, 280, 440)
    expect(bounds.min).toBeCloseTo(28)
    expect(bounds.max).toBeCloseTo(56)
    expect(bounds.min / 100 * 1000).toBeCloseTo(280)
    expect((100 - bounds.max) / 100 * 1000).toBeCloseTo(440)
  })

  it('scales minimums proportionally if the available space is too small', () => {
    const bounds = splitBounds(312, 260, 140)
    expect(bounds.min).toBeCloseTo(65)
    expect(bounds.max).toBeCloseTo(65)
    expect(splitBounds(0, 260, 140)).toEqual({ min: 0, max: 100 })
    expect(splitBounds(12, 260, 140)).toEqual({ min: 0, max: 100 })
  })

  it('measures late-mounted containers and clamps rendering without replacing saved preferences', () => {
    localStorage.setItem(WORKSPACE_LAYOUT_KEY, '{"statement":50,"editor":80}')
    let width = 1012
    let height = 1012
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      return { width: this.dataset.panel === 'workspace' ? width : 0, height: this.dataset.panel === 'column' ? height : 0 } as DOMRect
    })
    const observers: { callback: ResizeObserverCallback; observe: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> }[] = []
    vi.stubGlobal('ResizeObserver', class {
      observe = vi.fn()
      disconnect = vi.fn()
      unobserve = vi.fn()
      constructor(public callback: ResizeObserverCallback) { observers.push(this) }
    })
    let current!: ReturnType<typeof useWorkspaceLayout>
    function Harness({ mounted }: { mounted: boolean }) {
      current = useWorkspaceLayout()
      return mounted ? <div data-panel="workspace" ref={current.workspace.attach}><section data-panel="column" ref={current.column.attach} /></div> : null
    }

    const { rerender, unmount } = render(<Harness mounted={false} />)
    expect(observers).toHaveLength(0)
    rerender(<Harness mounted />)
    expect(observers).toHaveLength(2)
    expect(observers.every((observer) => observer.observe.mock.calls.length === 1)).toBe(true)
    expect(current.workspace.ref.current?.dataset.panel).toBe('workspace')
    expect(current.column.ref.current?.dataset.panel).toBe('column')
    expect(current.statement.value).toBe(50)
    expect(current.editor.value).toBe(80)

    width = 812
    height = 512
    act(() => observers.forEach((observer) => observer.callback([], observer as unknown as ResizeObserver)))
    expect(current.statement.value).toBeCloseTo(45)
    expect(current.editor.value).toBeCloseTo(72)
    act(() => vi.advanceTimersByTime(150))
    expect(JSON.parse(localStorage.getItem(WORKSPACE_LAYOUT_KEY)!)).toEqual({ statement: 50, editor: 80 })

    width = 1012
    height = 1012
    act(() => window.dispatchEvent(new Event('resize')))
    expect(current.statement.value).toBe(50)
    expect(current.editor.value).toBe(80)
    unmount()
    expect(observers.every((observer) => observer.disconnect.mock.calls.length === 1)).toBe(true)
  })
})
