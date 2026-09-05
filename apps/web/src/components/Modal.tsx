import { X } from 'lucide-react'
import { useEffect, useId, useRef, useState, type ReactNode } from 'react'

interface ModalProps {
  open: boolean
  title: string
  subtitle?: string
  children: ReactNode
  footer?: ReactNode
  onClose: () => void
  width?: string
}

const focusableSelector = 'button:not(:disabled), a[href], input:not(:disabled):not([type="hidden"]), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])'

export function Modal({ open, ...props }: ModalProps) {
  return open ? <ModalContent {...props} /> : null
}

function ModalContent({
  title,
  subtitle,
  children,
  footer,
  onClose,
  width = '520px',
}: Omit<ModalProps, 'open'>) {
  const titleId = useId()
  const subtitleId = useId()
  const dialogRef = useRef<HTMLElement>(null)
  const closeRef = useRef(onClose)
  // Capture the opener before child autoFocus runs during this mount.
  const [opener] = useState(() => document.activeElement instanceof HTMLElement ? document.activeElement : null)

  useEffect(() => {
    closeRef.current = onClose
  }, [onClose])

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    const focusableElements = () => Array.from(dialog.querySelectorAll<HTMLElement>(focusableSelector)).filter((element) => {
      const style = window.getComputedStyle(element)
      return !element.matches(':disabled') && !element.closest('[hidden], [inert]') && style.display !== 'none' && style.visibility !== 'hidden'
    })
    const focusFirst = () => (focusableElements()[0] ?? dialog).focus()
    if (!dialog.contains(document.activeElement)) {
      const input = focusableElements().find((element) => element.matches('input, textarea, select'))
      if (input) input.focus()
      else focusFirst()
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        closeRef.current()
      } else if (event.key === 'Tab') {
        const elements = focusableElements()
        const first = elements[0]
        const last = elements.at(-1)
        if (!first || !last) {
          event.preventDefault()
          dialog.focus()
        } else if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
          event.preventDefault()
          last.focus()
        } else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) {
          event.preventDefault()
          first.focus()
        }
      }
    }
    const keepFocusInside = (event: FocusEvent) => {
      if (event.target instanceof Node && !dialog.contains(event.target)) focusFirst()
    }
    window.addEventListener('keydown', onKeyDown, true)
    document.addEventListener('focusin', keepFocusInside)
    return () => {
      window.removeEventListener('keydown', onKeyDown, true)
      document.removeEventListener('focusin', keepFocusInside)
      document.body.style.overflow = previousOverflow
      if (opener?.isConnected) opener.focus()
    }
  }, [opener])

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section ref={dialogRef} className="modal" role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={subtitle ? subtitleId : undefined} tabIndex={-1} style={{ width }}>
        <header className="modal-header">
          <div>
            <h2 id={titleId}>{title}</h2>
            {subtitle && <p id={subtitleId}>{subtitle}</p>}
          </div>
          <button className="icon-button" type="button" aria-label="关闭对话框" onClick={onClose}><X size={18} /></button>
        </header>
        <div className="modal-body">{children}</div>
        {footer && <footer className="modal-footer">{footer}</footer>}
      </section>
    </div>
  )
}
