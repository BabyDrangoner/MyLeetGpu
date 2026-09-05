import { expect, test, type Locator, type Page } from '@playwright/test'
import { installMockApi } from './mock-api'

const layoutKey = 'myleetgpu:workspace-layout:v1'
const widthLabel = '调整题目与代码宽度'
const heightLabel = '调整代码与输出高度'

async function valueOf(separator: Locator) {
  return Number(await separator.getAttribute('aria-valuenow'))
}

async function drag(page: Page, separator: Locator, dx: number, dy: number) {
  await separator.scrollIntoViewIfNeeded()
  const bounds = await separator.boundingBox()
  if (!bounds) throw new Error('The resize separator is not visible')
  const x = bounds.x + bounds.width / 2
  const y = bounds.y + bounds.height / 2
  await page.mouse.move(x, y)
  await page.mouse.down()
  await page.mouse.move(x + dx, y + dy, { steps: 8 })
  await page.mouse.up()
}

async function dimension(panel: Locator, axis: 'width' | 'height') {
  const bounds = await panel.boundingBox()
  if (!bounds) throw new Error('The workspace panel is not visible')
  return bounds[axis]
}

async function expectNoPageOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1)
}

test.beforeEach(async ({ page }) => {
  await installMockApi(page)
  await page.goto('/problems/vector-addition')
  await expect(page.locator('.monaco-editor').first()).toBeVisible()
  await expect(page.getByRole('button', { name: '运行样例', exact: true })).toBeEnabled()
})

test('drags both dividers and restores the panel sizes after reload', async ({ page }) => {
  const horizontal = page.getByRole('separator', { name: widthLabel })
  const vertical = page.getByRole('separator', { name: heightLabel })
  const statement = page.locator('#problem-statement')
  const editor = page.locator('.editor-panel')
  const output = page.locator('.output-panel')
  const initialStatementWidth = await dimension(statement, 'width')
  const initialEditorWidth = await dimension(editor, 'width')
  const initialEditorHeight = await dimension(editor, 'height')
  const initialOutputHeight = await dimension(output, 'height')

  await expect(horizontal).toHaveAttribute('aria-orientation', 'vertical')
  await expect(vertical).toHaveAttribute('aria-orientation', 'horizontal')
  await drag(page, horizontal, 110, 0)
  await expect.poll(() => dimension(statement, 'width')).toBeGreaterThan(initialStatementWidth + 80)
  await expect.poll(() => dimension(editor, 'width')).toBeLessThan(initialEditorWidth - 80)
  await drag(page, vertical, 0, -90)
  await expect.poll(() => dimension(editor, 'height')).toBeLessThan(initialEditorHeight - 60)
  await expect.poll(() => dimension(output, 'height')).toBeGreaterThan(initialOutputHeight + 60)

  const resizedStatementWidth = await dimension(statement, 'width')
  const resizedEditorHeight = await dimension(editor, 'height')
  const resizedLayout = { statement: await valueOf(horizontal), editor: await valueOf(vertical) }
  await expect.poll(() => page.evaluate(({ key, expected }) => {
    const raw = localStorage.getItem(key)
    if (!raw) return Number.POSITIVE_INFINITY
    const saved = JSON.parse(raw) as { statement: number; editor: number }
    return Math.max(Math.abs(saved.statement - expected.statement), Math.abs(saved.editor - expected.editor))
  }, { key: layoutKey, expected: resizedLayout })).toBeLessThan(0.051)

  await page.reload()
  await expect(page.locator('.monaco-editor').first()).toBeVisible()
  await expect.poll(() => dimension(statement, 'width')).toBeCloseTo(resizedStatementWidth, 0)
  await expect.poll(() => dimension(editor, 'height')).toBeCloseTo(resizedEditorHeight, 0)
  await expectNoPageOverflow(page)
})

test('supports keyboard limits, double-click defaults and resetting both dimensions', async ({ page }) => {
  const horizontal = page.getByRole('separator', { name: widthLabel })
  const vertical = page.getByRole('separator', { name: heightLabel })

  for (const [separator, increaseKey, defaultValue] of [
    [horizontal, 'ArrowRight', 36],
    [vertical, 'ArrowDown', 73],
  ] as const) {
    const initial = await valueOf(separator)
    await separator.focus()
    await separator.press(increaseKey)
    await expect.poll(() => valueOf(separator)).toBeGreaterThan(initial)
    await separator.press('Home')
    await expect.poll(() => valueOf(separator)).toBeCloseTo(Number(await separator.getAttribute('aria-valuemin')), 0)
    await separator.press('End')
    await expect.poll(() => valueOf(separator)).toBeCloseTo(Number(await separator.getAttribute('aria-valuemax')), 0)
    await separator.dblclick()
    await expect.poll(() => valueOf(separator)).toBe(defaultValue)
    await expect(separator).toBeFocused()
  }

  await horizontal.press('ArrowRight')
  await vertical.press('ArrowUp')
  await expect.poll(() => valueOf(horizontal)).toBeGreaterThan(36)
  await expect.poll(() => valueOf(vertical)).toBeLessThan(73)
  await page.getByRole('button', { name: '重置布局', exact: true }).click()
  await expect.poll(() => valueOf(horizontal)).toBe(36)
  await expect.poll(() => valueOf(vertical)).toBe(73)
  await expect(page.getByRole('dialog')).toHaveCount(0)
})

test('bounds extreme drags and keeps the layout usable across desktop and mobile widths', async ({ page }) => {
  const horizontal = page.getByRole('separator', { name: widthLabel })
  const vertical = page.getByRole('separator', { name: heightLabel })
  for (const [separator, dx, dy, limit] of [
    [horizontal, 5_000, 0, 'aria-valuemax'],
    [horizontal, -5_000, 0, 'aria-valuemin'],
    [vertical, 0, 5_000, 'aria-valuemax'],
    [vertical, 0, -5_000, 'aria-valuemin'],
  ] as const) {
    await drag(page, separator, dx, dy)
    await expect.poll(() => valueOf(separator)).toBeCloseTo(Number(await separator.getAttribute(limit)), 0)
    await expectNoPageOverflow(page)
  }

  await page.setViewportSize({ width: 1024, height: 768 })
  await expect(horizontal).toBeVisible()
  await expect.poll(() => dimension(page.locator('#problem-statement'), 'width')).toBeGreaterThanOrEqual(279)
  await expect(page.getByRole('button', { name: '运行样例', exact: true })).toBeVisible()
  await expectNoPageOverflow(page)

  await page.setViewportSize({ width: 390, height: 844 })
  await expect(horizontal).toBeHidden()
  await expect(vertical).toBeVisible()
  await page.getByRole('button', { name: '重置布局', exact: true }).click()
  const editorHeight = await dimension(page.locator('.editor-panel'), 'height')
  await drag(page, vertical, 0, -75)
  await expect.poll(() => dimension(page.locator('.editor-panel'), 'height')).toBeLessThan(editorHeight - 45)
  await expectNoPageOverflow(page)
  await expect(page.getByRole('button', { name: '运行样例', exact: true })).toBeVisible()
})

test('preserves resized dimensions and the code draft through focus and output toggles', async ({ page }) => {
  const horizontal = page.getByRole('separator', { name: widthLabel })
  const vertical = page.getByRole('separator', { name: heightLabel })
  const editor = page.locator('.editor-panel')
  const marker = '// resize draft preserved'
  await page.locator('.monaco-editor .view-lines').click()
  await page.keyboard.press('ControlOrMeta+End')
  await page.keyboard.insertText(`\n${marker}`)
  await drag(page, horizontal, 90, 0)
  await drag(page, vertical, 0, -65)
  const statementWidth = await dimension(page.locator('#problem-statement'), 'width')
  const editorHeight = await dimension(editor, 'height')
  const layout = { statement: await valueOf(horizontal), editor: await valueOf(vertical) }

  await page.getByRole('button', { name: '专注编码' }).click()
  await expect(horizontal).toBeHidden()
  await expect(vertical).toBeVisible()
  await expect(page.locator('#problem-statement')).toBeHidden()
  await page.getByRole('button', { name: '收起任务与输出' }).click()
  await expect(vertical).toBeHidden()
  await expect.poll(() => dimension(editor, 'height')).toBeGreaterThan(editorHeight + 80)
  await page.getByRole('button', { name: '展开任务与输出' }).click()
  await page.getByRole('button', { name: '显示题目' }).click()
  await expect(horizontal).toBeVisible()
  await expect(vertical).toBeVisible()
  await expect.poll(() => dimension(page.locator('#problem-statement'), 'width')).toBeCloseTo(statementWidth, 0)
  await expect.poll(() => dimension(editor, 'height')).toBeCloseTo(editorHeight, 0)
  await expect.poll(() => valueOf(horizontal)).toBe(layout.statement)
  await expect.poll(() => valueOf(vertical)).toBe(layout.editor)
  await expect.poll(() => page.evaluate(() => {
    const raw = localStorage.getItem('myleetgpu:draft:vector-addition:cuda_cpp')
    return raw ? JSON.parse(raw).source as string : ''
  })).toContain(marker)

  await page.getByRole('button', { name: '重置布局', exact: true }).click()
  await page.reload()
  await expect(page.locator('.monaco-editor').first()).toBeVisible()
  await expect(page.locator('.monaco-editor .view-lines')).toContainText(marker)
  await expect.poll(() => valueOf(horizontal)).toBe(36)
  await expect.poll(() => valueOf(vertical)).toBe(73)
})
