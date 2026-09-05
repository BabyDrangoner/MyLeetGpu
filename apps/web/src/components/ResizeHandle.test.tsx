import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { createRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ResizeHandle } from './ResizeHandle'

afterEach(() => {
  cleanup()
  for (const element of [document.body, document.documentElement]) {
    element.style.removeProperty('cursor')
    element.style.removeProperty('user-select')
  }
  vi.restoreAllMocks()
})

function setup(axis: 'x' | 'y' = 'x', value = 40) {
  const onChange = vi.fn()
  const onReset = vi.fn()
  const containerRef = createRef<HTMLDivElement>()
  const result = render(
    <div ref={containerRef}>
      <ResizeHandle axis={axis} value={value} min={20} max={80} containerRef={containerRef} onChange={onChange} onReset={onReset} label="调整面板尺寸" controls="first-panel second-panel" />
    </div>,
  )
  const handle = screen.getByRole('separator', { name: '调整面板尺寸' })
  vi.spyOn(containerRef.current!, 'getBoundingClientRect').mockReturnValue({ width: 1012, height: 512 } as DOMRect)
  vi.spyOn(handle, 'getBoundingClientRect').mockReturnValue({ width: 12, height: 12 } as DOMRect)
  let capturedPointer: number | null = null
  handle.setPointerCapture = vi.fn((pointerId: number) => { capturedPointer = pointerId })
  handle.hasPointerCapture = vi.fn((pointerId: number) => capturedPointer === pointerId)
  handle.releasePointerCapture = vi.fn(() => { capturedPointer = null })
  return { ...result, handle, onChange, onReset, containerRef }
}

function pointer(target: Element, type: string, options: MouseEventInit & { pointerId?: number; isPrimary?: boolean } = {}) {
  const { pointerId = 1, isPrimary = true, ...mouseOptions } = options
  const event = new MouseEvent(type, { bubbles: true, cancelable: true, button: 0, ...mouseOptions })
  Object.defineProperties(event, { pointerId: { value: pointerId }, isPrimary: { value: isPrimary } })
  fireEvent(target, event)
}

describe('ResizeHandle', () => {
  it('exposes the controlled panels, orientation and current size', () => {
    const { handle } = setup('x', 42.34)
    expect(handle).toHaveAttribute('aria-controls', 'first-panel second-panel')
    expect(handle).toHaveAttribute('aria-orientation', 'vertical')
    expect(handle).toHaveAttribute('aria-valuemin', '20')
    expect(handle).toHaveAttribute('aria-valuemax', '80')
    expect(handle).toHaveAttribute('aria-valuenow', '42.34')
    expect(handle).toHaveAttribute('aria-valuetext', '42%')
    expect(handle).toHaveAttribute('tabindex', '0')
  })

  it('supports keyboard steps, limits and restoring the default size', () => {
    const { handle, onChange, onReset } = setup('x', 79)
    fireEvent.keyDown(handle, { key: 'ArrowLeft' })
    expect(onChange).toHaveBeenLastCalledWith(77)
    fireEvent.keyDown(handle, { key: 'ArrowLeft', shiftKey: true })
    expect(onChange).toHaveBeenLastCalledWith(69)
    fireEvent.keyDown(handle, { key: 'ArrowRight', shiftKey: true })
    expect(onChange).toHaveBeenLastCalledWith(80)
    fireEvent.keyDown(handle, { key: 'Home' })
    expect(onChange).toHaveBeenLastCalledWith(20)
    fireEvent.keyDown(handle, { key: 'End' })
    expect(onChange).toHaveBeenLastCalledWith(80)
    fireEvent.keyDown(handle, { key: 'Enter' })
    fireEvent.doubleClick(handle)
    expect(onReset).toHaveBeenCalledTimes(2)
  })

  it('uses vertical arrow keys only for horizontal separators', () => {
    const { handle, onChange } = setup('y', 21)
    expect(handle).toHaveAttribute('aria-orientation', 'horizontal')
    fireEvent.keyDown(handle, { key: 'ArrowLeft' })
    fireEvent.keyDown(handle, { key: 'ArrowRight' })
    expect(onChange).not.toHaveBeenCalled()
    fireEvent.keyDown(handle, { key: 'ArrowDown' })
    expect(onChange).toHaveBeenLastCalledWith(23)
    fireEvent.keyDown(handle, { key: 'ArrowUp', shiftKey: true })
    expect(onChange).toHaveBeenLastCalledWith(20)
  })

  it.each(['x', 'y'] as const)('resizes along %s using available panel space and clamps outside the container', (axis) => {
    const { handle, onChange } = setup(axis)
    pointer(handle, 'pointerdown', { clientX: 425, clientY: 225 })
    expect(handle.setPointerCapture).toHaveBeenCalledWith(1)
    expect(handle).toHaveFocus()
    expect(handle).toHaveClass('is-dragging')
    pointer(handle, 'pointermove', { clientX: 525, clientY: 275 })
    expect(onChange).toHaveBeenLastCalledWith(50)
    pointer(handle, 'pointermove', { clientX: 5000, clientY: 5000 })
    expect(onChange).toHaveBeenLastCalledWith(80)
    pointer(handle, 'pointermove', { clientX: -5000, clientY: -5000 })
    expect(onChange).toHaveBeenLastCalledWith(20)
  })

  it('ignores secondary buttons, non-primary pointers and other pointers during a drag', () => {
    const { handle, onChange } = setup()
    pointer(handle, 'pointerdown', { button: 2 })
    pointer(handle, 'pointerdown', { isPrimary: false })
    expect(handle.setPointerCapture).not.toHaveBeenCalled()
    expect(document.body.style.userSelect).toBe('')
    pointer(handle, 'pointerdown', { clientX: 400 })
    pointer(handle, 'pointermove', { pointerId: 2, clientX: 600 })
    pointer(handle, 'pointerup', { pointerId: 2 })
    expect(onChange).not.toHaveBeenCalled()
    expect(handle).toHaveClass('is-dragging')
    pointer(handle, 'pointermove', { clientX: 500 })
    expect(onChange).toHaveBeenLastCalledWith(50)
  })

  it.each(['pointerup', 'pointercancel', 'lostpointercapture', 'blur', 'resize', 'unmount'])('restores global styles and stops resizing on %s', (ending) => {
    document.body.style.setProperty('cursor', 'crosshair', 'important')
    document.body.style.userSelect = 'text'
    document.documentElement.style.cursor = 'wait'
    document.documentElement.style.userSelect = 'all'
    const { handle, onChange, unmount } = setup('y')
    pointer(handle, 'pointerdown', { clientY: 200 })
    expect(document.body.style.cursor).toBe('row-resize')
    expect(document.body.style.userSelect).toBe('none')
    expect(document.documentElement.style.cursor).toBe('row-resize')
    if (ending === 'unmount') unmount()
    else if (ending === 'blur' || ending === 'resize') fireEvent(window, new Event(ending))
    else pointer(handle, ending)
    expect(document.body.style.cursor).toBe('crosshair')
    expect(document.body.style.getPropertyPriority('cursor')).toBe('important')
    expect(document.body.style.userSelect).toBe('text')
    expect(document.documentElement.style.cursor).toBe('wait')
    expect(document.documentElement.style.userSelect).toBe('all')
    expect(handle.releasePointerCapture).toHaveBeenCalledWith(1)
    if (ending !== 'unmount') {
      expect(handle).not.toHaveClass('is-dragging')
      pointer(handle, 'pointermove', { clientY: 300 })
      expect(onChange).not.toHaveBeenCalled()
    }
  })

  it('does not start a drag without usable panel space or pointer capture', () => {
    const { handle, containerRef } = setup()
    vi.mocked(containerRef.current!.getBoundingClientRect).mockReturnValue({ width: 12, height: 512 } as DOMRect)
    pointer(handle, 'pointerdown')
    expect(handle.setPointerCapture).not.toHaveBeenCalled()
    expect(document.body.style.userSelect).toBe('')
    vi.mocked(containerRef.current!.getBoundingClientRect).mockReturnValue({ width: 1012, height: 512 } as DOMRect)
    vi.mocked(handle.setPointerCapture).mockImplementation(() => { throw new DOMException('Pointer no longer active') })
    pointer(handle, 'pointerdown')
    expect(handle).not.toHaveClass('is-dragging')
    expect(document.body.style.userSelect).toBe('')
  })
})
