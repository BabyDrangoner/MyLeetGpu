import { expect, test, type Page } from '@playwright/test'
import { installMockApi } from './mock-api'

async function expectNoPageOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1)
}

test('filters the compact catalog and persists the theme across navigation and reload', async ({ page }, testInfo) => {
  await installMockApi(page)
  await page.goto('/problems')
  await expect(page.locator('.problem-row')).toHaveCount(10)
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')
  await page.screenshot({ path: testInfo.outputPath('catalog-light.png'), fullPage: true, animations: 'disabled' })
  await page.getByRole('button', { name: /PyTorch 题/ }).click()
  await expect(page.locator('.problem-row')).toHaveCount(2)
  await page.getByLabel('按难度筛选').selectOption('困难')
  await expect(page.locator('.problem-row')).toHaveCount(1)
  await expect(page.getByRole('link', { name: /分组查询自注意力/ })).toBeVisible()
  await page.getByRole('button', { name: '清除筛选' }).click()
  await page.getByRole('button', { name: '切换到深色模式' }).click()
  await page.reload()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  await expect(page.locator('.problem-row')).toHaveCount(10)
  await page.screenshot({ path: testInfo.outputPath('catalog-dark.png'), fullPage: true, animations: 'disabled' })
  await page.getByRole('link', { name: /向量逐元素相加/ }).click()
  await expect(page.locator('.monaco-editor').first()).toBeVisible()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  await expectNoPageOverflow(page)
})

test('focus and output controls preserve editor content and the run shortcut uses the latest code', async ({ page }, testInfo) => {
  const state = await installMockApi(page)
  await page.goto('/problems/vector-addition')
  await expect(page.locator('.monaco-editor').first()).toBeVisible()
  await expect(page.getByRole('button', { name: '运行样例', exact: true })).toBeEnabled()
  await page.screenshot({ path: testInfo.outputPath('workspace-light.png'), fullPage: true, animations: 'disabled' })
  const initialEditor = await page.locator('.editor-panel').boundingBox()
  await page.getByRole('button', { name: '专注编码' }).click()
  await expect(page.getByRole('button', { name: '显示题目' })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.locator('#problem-statement')).toBeHidden()
  await expect.poll(async () => (await page.locator('.editor-panel').boundingBox())?.width ?? 0).toBeGreaterThan((initialEditor?.width ?? 0) * 1.2)
  await page.getByRole('button', { name: '收起任务与输出' }).click()
  await expect(page.locator('#workspace-output')).toBeHidden()
  await page.locator('.monaco-editor .view-lines').click()
  await page.keyboard.press('ControlOrMeta+End')
  await page.keyboard.insertText('\n// interface draft preserved')
  await page.getByRole('button', { name: '切换到深色模式' }).click()
  await page.screenshot({ path: testInfo.outputPath('workspace-focus-dark.png'), fullPage: true, animations: 'disabled' })
  await page.locator('.monaco-editor .view-lines').click()
  await page.keyboard.press('ControlOrMeta+Enter')
  await expect.poll(() => state.submittedJobs.length).toBe(1)
  expect(state.submittedJobs[0]).toMatchObject({ action: 'run', language: 'cuda_cpp' })
  expect(state.submittedJobs[0].source).toContain('// interface draft preserved')
  await expect(page.locator('#workspace-output')).toBeVisible()
  await expect(page.getByText('公开样例运行超过限制')).toBeVisible()
  await expectNoPageOverflow(page)
})

for (const viewport of [{ width: 1024, height: 768 }, { width: 390, height: 844 }, { width: 320, height: 640 }]) {
  test(`keeps navigation, coding and version controls reachable at ${viewport.width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport)
    await installMockApi(page, { versions: true, comparable: true })
    for (const [name, route, ready] of [
      ['catalog', '/problems', '.problem-row'],
      ['workspace', '/problems/vector-addition', '.monaco-editor'],
      ['environment', '/environment', '.environment-hero'],
      ['versions', '/problems/vector-addition/versions', '.monaco-diff-editor'],
    ]) {
      await page.goto(route)
      await expect(page.locator(ready).first()).toBeVisible()
      if (name === 'versions') await expect(page.getByText('1.39×').first()).toBeVisible()
      await expectNoPageOverflow(page)
      await page.screenshot({ path: testInfo.outputPath(`${name}-${viewport.width}.png`), fullPage: true, animations: 'disabled' })
    }
    await page.goto('/problems/vector-addition')
    await page.getByRole('button', { name: '保存为性能版本' }).click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    const bounds = await dialog.boundingBox()
    expect(bounds?.x).toBeGreaterThanOrEqual(0)
    expect((bounds?.x ?? 0) + (bounds?.width ?? 0)).toBeLessThanOrEqual(viewport.width)
    expect((bounds?.y ?? 0) + (bounds?.height ?? 0)).toBeLessThanOrEqual(viewport.height)
    await expect(page.getByRole('button', { name: '验证、测速并保存' })).toBeInViewport()
    await page.screenshot({ path: testInfo.outputPath(`save-dialog-${viewport.width}.png`), fullPage: true, animations: 'disabled' })
    await page.getByRole('button', { name: '取消', exact: true }).focus()
    await page.keyboard.press('Shift+Tab')
    expect(await dialog.evaluate((element) => element.contains(document.activeElement))).toBe(true)
    await page.keyboard.press('Escape')
    await expect(dialog).toBeHidden()
    await expect(page.getByRole('button', { name: '保存为性能版本' })).toBeFocused()
  })
}

test('shows environment and version data clearly in both themes', async ({ page }, testInfo) => {
  await installMockApi(page, { versions: true, comparable: true })
  for (const theme of ['light', 'dark']) {
    await page.goto('/environment')
    if (theme === 'dark') await page.getByRole('button', { name: '切换到深色模式' }).click()
    await expect(page.getByText('环境就绪', { exact: true })).toBeVisible()
    await page.screenshot({ path: testInfo.outputPath(`environment-${theme}.png`), fullPage: true, animations: 'disabled' })
    await page.goto('/problems/vector-addition/versions')
    await expect(page.locator('.monaco-diff-editor')).toBeVisible()
    await expect(page.getByText('可直接比较')).toBeVisible()
    await expect(page.getByText('1.39×').first()).toBeVisible()
    await page.screenshot({ path: testInfo.outputPath(`versions-${theme}.png`), fullPage: true, animations: 'disabled' })
    await expectNoPageOverflow(page)
  }
})
