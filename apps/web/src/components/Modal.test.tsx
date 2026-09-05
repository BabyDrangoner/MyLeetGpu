import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { StrictMode, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Modal } from './Modal'

afterEach(() => {
  cleanup()
  document.body.style.overflow = ''
})

describe('Modal keyboard interaction', () => {
  it('keeps tab navigation inside the dialog and skips disabled controls', async () => {
    const user = userEvent.setup()
    render(
      <Modal open title="保存版本" subtitle="保存当前代码快照" onClose={vi.fn()} footer={<><button disabled>暂不可用</button><button>保存</button></>}>
        <label>名称<input /></label>
      </Modal>,
    )

    expect(screen.getByRole('dialog', { name: '保存版本' })).toHaveAccessibleDescription('保存当前代码快照')
    expect(screen.getByRole('textbox', { name: '名称' })).toHaveFocus()
    await user.tab()
    expect(screen.getByRole('button', { name: '保存' })).toHaveFocus()
    await user.tab()
    expect(screen.getByRole('button', { name: '关闭对话框' })).toHaveFocus()
    await user.tab({ shift: true })
    expect(screen.getByRole('button', { name: '保存' })).toHaveFocus()
  })

  it('preserves the active field on rerender and calls the latest Escape handler', async () => {
    const user = userEvent.setup()
    const firstClose = vi.fn()
    const nextClose = vi.fn()
    const dialog = (onClose: () => void) => (
      <Modal open title="编辑版本" onClose={onClose}>
        <label>名称<input /></label>
        <label>备注<textarea /></label>
      </Modal>
    )
    const { rerender } = render(dialog(firstClose))
    await user.click(screen.getByRole('textbox', { name: '备注' }))
    rerender(dialog(nextClose))
    expect(screen.getByRole('textbox', { name: '备注' })).toHaveFocus()

    await user.keyboard('{Escape}')
    expect(nextClose).toHaveBeenCalledOnce()
    expect(firstClose).not.toHaveBeenCalled()
  })

  it('restores the opener and existing body scroll style after closing an autofocus form', async () => {
    function Example() {
      const [open, setOpen] = useState(false)
      return <>
        <button onClick={() => setOpen(true)}>打开</button>
        <Modal open={open} title="保存版本" onClose={() => setOpen(false)}>
          <label>名称<input autoFocus /></label>
        </Modal>
      </>
    }
    const user = userEvent.setup()
    document.body.style.overflow = 'scroll'
    render(<StrictMode><Example /></StrictMode>)
    const opener = screen.getByRole('button', { name: '打开' })
    await user.click(opener)
    expect(screen.getByRole('textbox', { name: '名称' })).toHaveFocus()
    expect(document.body.style.overflow).toBe('hidden')

    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(opener).toHaveFocus()
    expect(document.body.style.overflow).toBe('scroll')
  })
})
