import { expect, test } from '@playwright/test'
import { installMockApi, starterSource, torchStarterSource, tritonStarterSource } from './mock-api'

test('loads all eight problems and opens the CUDA workspace', async ({ page }) => {
  await installMockApi(page)
  await page.goto('/problems')

  await expect(page.getByRole('heading', { name: '把想法变成可靠、快速的实现' })).toBeVisible()
  await expect(page.getByRole('link', { name: /向量逐元素相加/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /行主序矩阵转置/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /单精度向量求和归约/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /单精度向量最大值归约/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /逐行 Softmax/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /行主序矩阵乘法/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /多头注意力/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /分组查询注意力/ })).toBeVisible()
  await page.getByRole('link', { name: /向量逐元素相加/ }).click()

  await expect(page.getByRole('heading', { name: '向量逐元素相加', level: 1 }).first()).toBeVisible()
  await expect(page.getByText('solution.cu')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Triton (Python)' })).toBeVisible()
  await expect(page.getByText('等待任务')).toBeVisible()
  await page.getByRole('button', { name: '测量协议' }).click()
  await expect(page.getByText('64K / 1M')).toBeVisible()
  await expect(page.getByText('median；同时记录 p95、min 与波动')).toBeVisible()
})

test('classifies compile errors, wrong answers and timeouts without creating versions', async ({ page }) => {
  const state = await installMockApi(page)
  await page.goto('/problems/vector-addition')

  await page.getByRole('button', { name: '编译' }).click()
  await expect(page.getByText('NVCC 编译失败')).toBeVisible()
  await expect(page.getByText('solution.cu:7: error: expected a semicolon')).toBeVisible()
  await page.getByRole('button', { name: '清空' }).click()

  await page.getByRole('button', { name: '运行样例' }).click()
  await expect(page.getByText('已超时', { exact: true })).toBeVisible()
  await expect(page.getByText('公开样例运行超过限制')).toBeVisible()
  await page.getByRole('button', { name: '清空' }).click()

  await page.getByRole('button', { name: '完整验证' }).click()
  await expect(page.getByText('公开样例 1')).toBeVisible()
  await expect(page.getByText(/wrong_answer/)).toBeVisible()
  expect(state.versions).toHaveLength(0)
  expect(state.submittedJobs.map((job) => job.action)).toEqual(['compile', 'run', 'validate'])
})

test('saves the click-time snapshot explicitly and keeps it after refresh', async ({ page }) => {
  const state = await installMockApi(page)
  await page.goto('/problems/vector-addition')
  await page.getByRole('button', { name: '保存为性能版本' }).click()

  await expect(page.getByText('已锁定点击时的代码快照')).toBeVisible()
  await page.getByPlaceholder('例如：共享内存分块 v2').fill('首个稳定版本')
  await page.getByPlaceholder('记录这次优化的思路、取舍或待验证假设…').fill('E2E 保存语义')
  await page.getByRole('button', { name: '验证、测速并保存' }).click()
  await expect(page.getByText('性能版本已保存')).toBeVisible()
  await expect(page.getByRole('link', { name: /性能版本 1/ })).toBeVisible()

  expect(state.submittedJobs).toHaveLength(1)
  expect(state.submittedJobs[0]).toMatchObject({ action: 'save_version', language: 'cuda_cpp', version_name: '首个稳定版本', source: starterSource })
  await page.reload()
  await expect(page.getByRole('link', { name: /性能版本 1/ })).toBeVisible()
  await page.getByRole('link', { name: /性能版本 1/ }).click()
  await expect(page.getByText('首个稳定版本')).toBeVisible()
})

test('switches to Triton with a language-scoped starter, URL and job payload', async ({ page }) => {
  const state = await installMockApi(page)
  await page.goto('/problems/vector-addition')

  await page.getByRole('button', { name: 'Triton (Python)' }).click()
  await expect(page).toHaveURL(/language=triton_python/)
  await expect(page.getByText('solution.py')).toBeVisible()
  await expect(page.locator('.signature-block code')).toHaveText('def solve(a: torch.Tensor, b: torch.Tensor, output: torch.Tensor, n: int) -> None')
  await page.getByRole('button', { name: '运行样例' }).click()

  expect(state.submittedJobs.at(-1)).toMatchObject({ action: 'run', language: 'triton_python', source: tritonStarterSource })
  await page.reload()
  await expect(page).toHaveURL(/language=triton_python/)
  await expect(page.getByText('solution.py')).toBeVisible()

  await page.getByRole('button', { name: 'CUDA C++' }).click()
  await expect(page).toHaveURL(/language=cuda_cpp/)
  await expect(page.getByText('solution.cu')).toBeVisible()
})

test('opens a torch-only attention problem with a language-scoped Python session', async ({ page }) => {
  const state = await installMockApi(page)
  await page.goto('/problems/multi-head-attention?language=cuda_cpp')

  await expect(page).toHaveURL(/language=torch_python/)
  await expect(page.getByRole('heading', { name: '多头注意力', level: 1 }).first()).toBeVisible()
  await expect(page.getByText('solution.py')).toBeVisible()
  await expect(page.getByRole('button', { name: 'PyTorch (Python)' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'CUDA C++' })).toHaveCount(0)
  await page.getByRole('button', { name: '代码检查' }).click()
  await expect(page.getByText('PyTorch 代码检查失败')).toBeVisible()
  expect(state.submittedJobs.at(-1)).toMatchObject({ action: 'compile', language: 'torch_python', source: torchStarterSource })
})

test('shows the independently probed Triton toolchain environment', async ({ page }) => {
  await installMockApi(page)
  await page.goto('/environment')

  await expect(page.getByText('NVCC', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Triton (Python)' }).click()
  await expect(page.getByText('Python / PyTorch / Triton')).toBeVisible()
  await expect(page.getByText('3.11.10 / 2.5.1+cu124 / 3.2.0')).toBeVisible()

  await page.getByRole('button', { name: 'PyTorch (Python)' }).click()
  await expect(page.getByText('Python / PyTorch / Torch CUDA')).toBeVisible()
  await expect(page.getByText('3.11.10 / 2.5.1+cu124 / 12.6')).toBeVisible()
})
