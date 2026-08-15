import { expect, test } from '@playwright/test'
import { installMockApi, starterSource } from './mock-api'

test('loads three original problems and opens the CUDA workspace', async ({ page }) => {
  await installMockApi(page)
  await page.goto('/problems')

  await expect(page.getByRole('heading', { name: '把想法变成更快的 Kernel' })).toBeVisible()
  await expect(page.getByRole('link', { name: /向量逐元素相加/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /行主序矩阵转置/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /单精度向量求和归约/ })).toBeVisible()
  await page.getByRole('link', { name: /向量逐元素相加/ }).click()

  await expect(page.getByRole('heading', { name: '向量逐元素相加', level: 1 }).first()).toBeVisible()
  await expect(page.getByText('solution.cu')).toBeVisible()
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
  expect(state.submittedJobs[0]).toMatchObject({ action: 'save_version', version_name: '首个稳定版本', source: starterSource })
  await page.reload()
  await expect(page.getByRole('link', { name: /性能版本 1/ })).toBeVisible()
  await page.getByRole('link', { name: /性能版本 1/ }).click()
  await expect(page.getByText('首个稳定版本')).toBeVisible()
})
