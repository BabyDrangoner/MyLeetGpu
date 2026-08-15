import { expect, test } from '@playwright/test'
import { installMockApi } from './mock-api'

test('compares two versions against a baseline and renders Monaco Diff', async ({ page }) => {
  await installMockApi(page, { versions: true, comparable: true })
  await page.goto('/problems/vector-addition/versions')

  await expect(page.getByText('可直接比较')).toBeVisible()
  await expect(page.getByText('1.39×').first()).toBeVisible()
  await expect(page.getByText('20 样本').first()).toBeVisible()
  await expect(page.getByText('CV 3.0%').first()).toBeVisible()
  await expect(page.getByRole('heading', { name: /代码快照 Diff/ })).toBeVisible()
  await expect(page.locator('.monaco-diff-editor')).toBeVisible()
  await expect(page.getByText('协议版本').first()).toBeVisible()
  await expect(page.getByText('--std=c++17 -O3').first()).toBeVisible()
  await expect(page.getByText('591.74').first()).toBeVisible()
  await expect(page.getByText('sha256:e2e').first()).toBeVisible()
})

test('refuses unified speedup when environment fingerprints differ', async ({ page }) => {
  await installMockApi(page, { versions: true, comparable: false })
  await page.goto('/problems/vector-addition/versions')

  await expect(page.getByText('不可直接比较')).toBeVisible()
  await expect(page.getByText(/GPU\/CUDA 环境指纹不同/)).toBeVisible()
  await expect(page.getByText('不计算 speedup').first()).toBeVisible()
  await expect(page.getByText('环境不一致')).toBeVisible()
})

test('edits metadata and requires the destructive second confirmation', async ({ page }) => {
  const state = await installMockApi(page, { versions: true, comparable: true })
  await page.goto('/problems/vector-addition/versions')

  await page.getByRole('button', { name: '编辑 直接加载' }).click()
  await page.locator('.modal input').fill('基线实现 v2')
  await page.locator('.modal textarea').fill('更新后的备注')
  await page.getByRole('button', { name: '保存修改' }).click()
  await expect(page.locator('.version-row').filter({ hasText: '更新后的备注' }).getByText('基线实现 v2', { exact: true })).toBeVisible()

  await page.getByRole('button', { name: '删除 基线实现 v2' }).click()
  await expect(page.getByRole('heading', { name: '确认删除性能版本？' })).toBeVisible()
  await expect(page.getByText('这是第二次确认。删除后，关联的 benchmark 历史也将不可恢复。')).toBeVisible()
  await page.getByRole('button', { name: '确认删除' }).click()
  await expect(page.locator('.version-row').filter({ hasText: '基线实现 v2' })).toHaveCount(0)
  expect(state.deleteConfirmed).toBe(true)
})
