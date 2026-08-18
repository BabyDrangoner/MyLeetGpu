import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { CodeDiff, CodeEditor } from './CodeEditor'

vi.mock('../monaco', () => ({}))

describe('language-aware code editors', () => {
  it('uses a Python model and .py path for Triton', () => {
    render(<CodeEditor value="# triton" language="triton_python" problemId="vector-addition" />)
    const editor = screen.getByLabelText('Triton Python 代码编辑器')
    expect(editor).toHaveAttribute('data-editor-language', 'python')
    expect(editor).toHaveAttribute('data-editor-path', 'solution.py')
  })

  it('uses C++ highlighting for CUDA diffs', () => {
    const { container } = render(<CodeDiff original="// old" modified="// new" language="cuda_cpp" />)
    expect(container.querySelector('.test-diff')).toHaveAttribute('data-editor-language', 'cpp')
  })
})
