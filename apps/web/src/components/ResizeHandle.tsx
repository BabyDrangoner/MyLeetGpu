import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent, type RefObject } from 'react'

interface ResizeHandleProps {
  axis: 'x' | 'y'
  value: number
  min: number
  max: number
  containerRef: RefObject<HTMLElement>
  onChange: (percent: number) => void
  onReset: () => void
  label: string
  controls: string
}

interface DragState {
  element: HTMLDivElement
  pointerId: number
  startPosition: number
  startValue: number
  availableSize: number
  restoreStyles: () => void
}

function setDragStyles(axis: 'x' | 'y') {
  const properties = ['cursor', 'user-select'] as const
  const originals = [document.documentElement, document.body].map((element) => ({
    element,
    properties: properties.map((property) => ({
      property,
      value: element.style.getPropertyValue(property),
      priority: element.style.getPropertyPriority(property),
    })),
  }))

  for (const { element } of originals) {
    element.style.setProperty('cursor', axis === 'x' ? 'col-resize' : 'row-resize')
    element.style.setProperty('user-select', 'none')
  }

  return () => {
    for (const { element, properties: previousProperties } of originals) {
      for (const { property, value, priority } of previousProperties) {
        if (value) element.style.setProperty(property, value, priority)
        else element.style.removeProperty(property)
      }
    }
  }
}

export function ResizeHandle({ axis, value, min, max, containerRef, onChange, onReset, label, controls }: ResizeHandleProps) {
  const dragRef = useRef<DragState | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const clamp = (next: number) => Math.max(min, Math.min(max, next))

  const finishResize = useCallback((updateState = true) => {
    const drag = dragRef.current
    if (!drag) return
    dragRef.current = null
    drag.restoreStyles()
    if (drag.element.hasPointerCapture(drag.pointerId)) {
      drag.element.releasePointerCapture(drag.pointerId)
    }
    if (updateState) setIsDragging(false)
  }, [])

  useEffect(() => {
    const onWindowChange = () => finishResize()
    window.addEventListener('blur', onWindowChange)
    window.addEventListener('resize', onWindowChange)
    return () => {
      window.removeEventListener('blur', onWindowChange)
      window.removeEventListener('resize', onWindowChange)
      finishResize(false)
    }
  }, [finishResize])

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || !event.isPrimary || dragRef.current) return
    const container = containerRef.current
    if (!container) return
    const containerBounds = container.getBoundingClientRect()
    const handleBounds = event.currentTarget.getBoundingClientRect()
    const availableSize = axis === 'x'
      ? containerBounds.width - handleBounds.width
      : containerBounds.height - handleBounds.height
    if (availableSize <= 0) return

    // Capture keeps drag events on the separator when crossing the code editor.
    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      return
    }
    event.preventDefault()
    event.currentTarget.focus({ preventScroll: true })
    dragRef.current = {
      element: event.currentTarget,
      pointerId: event.pointerId,
      startPosition: axis === 'x' ? event.clientX : event.clientY,
      startValue: clamp(value),
      availableSize,
      restoreStyles: setDragStyles(axis),
    }
    setIsDragging(true)
  }

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current
    if (!drag || event.pointerId !== drag.pointerId) return
    event.preventDefault()
    const position = axis === 'x' ? event.clientX : event.clientY
    onChange(clamp(drag.startValue + (position - drag.startPosition) / drag.availableSize * 100))
  }

  const onPointerEnd = (event: PointerEvent<HTMLDivElement>) => {
    if (event.pointerId === dragRef.current?.pointerId) finishResize()
  }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 10 : 2
    const decreaseKey = axis === 'x' ? 'ArrowLeft' : 'ArrowUp'
    const increaseKey = axis === 'x' ? 'ArrowRight' : 'ArrowDown'
    let next: number
    if (event.key === decreaseKey) next = value - step
    else if (event.key === increaseKey) next = value + step
    else if (event.key === 'Home') next = min
    else if (event.key === 'End') next = max
    else if (event.key === 'Enter') {
      event.preventDefault()
      finishResize()
      onReset()
      return
    } else return
    event.preventDefault()
    onChange(clamp(next))
  }

  return (
    <div
      className={`resize-handle resize-handle-${axis}${isDragging ? ' is-dragging' : ''}`}
      role="separator"
      tabIndex={0}
      aria-label={label}
      aria-controls={controls}
      aria-orientation={axis === 'x' ? 'vertical' : 'horizontal'}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={clamp(value)}
      aria-valuetext={`${Math.round(clamp(value))}%`}
      title={`${label}：拖拽或使用方向键调整，Shift 加速，双击或 Enter 恢复默认`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerEnd}
      onPointerCancel={onPointerEnd}
      onLostPointerCapture={onPointerEnd}
      onKeyDown={onKeyDown}
      onDoubleClick={() => {
        finishResize()
        onReset()
      }}
    />
  )
}
