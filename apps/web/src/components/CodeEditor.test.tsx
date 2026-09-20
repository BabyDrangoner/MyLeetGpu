import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { CodeDiff, CodeEditor } from './CodeEditor'

vi.mock('../monaco', () => ({}))
afterEach(cleanup)

describe('language-aware code editors', () => {
  it('uses a Python model and .py path for Triton', () => {
    render(<CodeEditor value="# triton" language="triton_python" problemId="vector-addition" />)
    const editor = screen.getByLabelText('Triton (Python) 代码编辑器')
    expect(editor).toHaveAttribute('data-editor-language', 'python')
    expect(editor).toHaveAttribute('data-editor-path', '/problems/vector-addition/triton_python/solution.py')
  })

  it('isolates PyTorch and Triton in separate Python models', () => {
    render(
      <>
        <CodeEditor value="# triton" language="triton_python" problemId="attention" />
        <CodeEditor value="# torch" language="torch_python" problemId="attention" />
      </>,
    )
    const triton = screen.getByLabelText('Triton (Python) 代码编辑器')
    const torch = screen.getByLabelText('PyTorch (Python) 代码编辑器')
    expect(torch).toHaveAttribute('data-editor-language', 'python')
    expect(torch).toHaveAttribute('data-editor-path', '/problems/attention/torch_python/solution.py')
    expect(torch.getAttribute('data-editor-path')).not.toBe(triton.getAttribute('data-editor-path'))
  })

  it('uses C++ highlighting for CUDA diffs', () => {
    const { container } = render(<CodeDiff original="// old" modified="// new" language="cuda_cpp" />)
    expect(container.querySelector('.test-diff')).toHaveAttribute('data-editor-language', 'cpp')
  })

  it('uses .cpp and separate Python model paths for CPU implementations', () => {
    render(<><CodeEditor value="// cpp" language="cpp" problemId="online-softmax" /><CodeEditor value="# python" language="python" problemId="online-softmax" /></>)
    expect(screen.getByLabelText('C++ 代码编辑器')).toHaveAttribute('data-editor-path', '/problems/online-softmax/cpp/solution.cpp')
    expect(screen.getByLabelText('C++ 代码编辑器')).toHaveAttribute('data-editor-language', 'cpp')
    expect(screen.getByLabelText('Python 代码编辑器')).toHaveAttribute('data-editor-path', '/problems/online-softmax/python/solution.py')
    expect(screen.getByLabelText('Python 代码编辑器')).toHaveAttribute('data-editor-language', 'python')
  })
})
