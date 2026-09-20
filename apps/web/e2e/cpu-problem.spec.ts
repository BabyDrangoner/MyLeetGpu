import { expect, test } from '@playwright/test'
import { installMockApi } from './mock-api'

test('opens a CPU question, switches C++ and Python, and probes without changing GPU settings', async ({ page }, testInfo) => {
  const state = await installMockApi(page)
  state.executionSettings.target = 'colab'
  state.executionSettings.colab_acknowledged = false
  const summary = { slug: 'online-softmax', title: 'Online Softmax', difficulty: 'medium', revision: '1', summary: 'CPU 在线归一化', languages: ['cpp', 'python'] }
  const requests: Array<Record<string, unknown>> = []
  await page.route('**/api/problems', (route) => route.fulfill({ json: { items: [summary] } }))
  await page.route('**/api/problems/online-softmax', (route) => route.fulfill({ json: {
    ...summary, default_language: 'cpp', statement_markdown: '## Online Softmax\n\n实现在线归一化。',
    implementations: {
      cpp: { starter_code: '// CPU C++ starter', source_suffix: '.cpp' },
      python: { starter_code: '# CPU Python starter', source_suffix: '.py' },
    },
  } }))
  await page.route('**/api/jobs', async (route) => {
    const body = route.request().postDataJSON() as Record<string, unknown>
    requests.push(body)
    await route.fulfill({ json: { ...body, id: `cpu-${requests.length}`, status: 'succeeded', execution_target: 'cpu', result: { message: 'CPU 样例通过' } } })
  })
  const cpuSnapshot = { execution_target: 'cpu', healthy: true, toolchain: { cpu_name: 'Test CPU', platform: 'Linux', architecture: 'x86_64', compiler_version: 'clang 17', python_version: '3.12.8' } }
  await page.route('**/api/environment?*', (route) => {
    const language = new URL(route.request().url()).searchParams.get('language')
    return language === 'cpp' || language === 'python' ? route.fulfill({ json: { ...cpuSnapshot, backend: language } }) : route.fallback()
  })
  await page.route('**/api/execution-settings/probe', (route) => {
    const body = route.request().postDataJSON() as { target: string; language: string }
    state.probes.push(body)
    return route.fulfill({ json: { ...cpuSnapshot, backend: body.language } })
  })

  await page.goto('/problems')
  await page.getByRole('button', { name: /CPU 算法题/ }).click()
  await page.getByRole('link', { name: /Online Softmax/ }).click()
  await expect(page.getByText('solution.cpp', { exact: true })).toBeVisible()
  await expect(page.getByRole('link', { name: '执行：本地 CPU', exact: true })).toBeVisible()
  await expect(page.getByText(/不使用安全沙箱/)).toBeVisible()
  await page.getByRole('button', { name: '运行样例', exact: true }).click()
  await expect(page.locator('.output-status')).toHaveText('本地 CPU')
  await page.getByRole('button', { name: 'Python', exact: true }).click()
  await expect(page.getByText('solution.py', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: '代码检查', exact: true }).click()
  await expect.poll(() => requests.length).toBe(2)
  expect(requests[0]).toMatchObject({ language: 'cpp', action: 'run' })
  expect(requests[1]).toMatchObject({ language: 'python', action: 'compile' })
  await expect(page.locator('.output-status')).toHaveText('本地 CPU')
  await expect(page.getByText('CPU 样例通过', { exact: true })).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('cpu-workspace.png'), fullPage: true, animations: 'disabled' })

  await page.getByRole('link', { name: '执行：本地 CPU', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Python', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByText('Test CPU')).toBeVisible()
  await expect(page.getByText('CUDA Runtime', { exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: '保存设置', exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: '探测 CPU 环境', exact: true }).click()
  await expect.poll(() => state.probes).toEqual([{ target: 'cpu', language: 'python' }])
  expect(state.executionSettings.target).toBe('colab')
  expect(state.executionSettings.colab_acknowledged).toBe(false)
  await page.setViewportSize({ width: 320, height: 740 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1)
  await page.screenshot({ path: testInfo.outputPath('cpu-environment-mobile.png'), fullPage: true, animations: 'disabled' })
})
