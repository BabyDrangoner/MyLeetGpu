import { BookOpenText, Code2, Cpu, Github, Moon, Sun } from 'lucide-react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { api } from '../api/client'
import { useAsync } from '../hooks/useAsync'
import { useTheme } from '../hooks/useTheme'
import { ExecutionSettingsProvider, executionTargetLabel, useExecutionSettings } from '../hooks/useExecutionSettings'

export function AppShell() {
  return <ExecutionSettingsProvider><AppShellContent /></ExecutionSettingsProvider>
}

function AppShellContent() {
  const execution = useExecutionSettings()
  const environment = useAsync(() => api.environment(), [execution?.settings?.target, execution?.settings?.updated_at])
  const { pathname } = useLocation()
  const { theme, toggleTheme } = useTheme()
  const healthy = environment.data?.healthy ?? environment.data?.status === 'healthy'
  const hostname = window.location.hostname
  const localMode = hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1' || hostname === '[::1]'
  const locationLabel = pathname === '/environment'
    ? '运行环境'
    : pathname.endsWith('/versions')
      ? '题目 / 版本记录'
      : pathname.startsWith('/problems/')
        ? '题目 / 编程工作台'
        : '题目库'
  const environmentLabel = environment.loading ? '检查 GPU 环境中' : healthy ? 'GPU 环境就绪' : '查看 GPU 环境状态'
  const themeLabel = theme === 'light' ? '切换到深色模式' : '切换到浅色模式'

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">跳到主要内容</a>
      <aside className="sidebar">
        <Link className="brand-block" to="/problems" aria-label="MyLeetGpu 题目库" title="MyLeetGpu">
          <span className="brand-symbol" aria-hidden="true"><Code2 size={22} strokeWidth={1.7} /></span>
          <span className="brand-copy"><span className="brand-name">MyLeetGpu</span></span>
        </Link>

        <nav className="main-nav" aria-label="主导航">
          <NavLink to="/problems" aria-label="题目工作台" title="题目工作台" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
            <BookOpenText size={18} aria-hidden="true" />
            <span className="nav-label">题目工作台</span>
          </NavLink>
          <NavLink to="/environment" aria-label="运行环境" title="运行环境" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
            <Cpu size={18} aria-hidden="true" />
            <span className="nav-label">运行环境</span>
          </NavLink>
        </nav>

        <div className="sidebar-spacer" />
        <NavLink to="/environment" className="environment-mini" aria-label={environmentLabel} title={environmentLabel}>
          <div className="environment-mini-head">
            <span aria-hidden="true" className={`health-dot ${environment.loading ? 'checking' : healthy ? 'healthy' : 'unhealthy'}`} />
            <span>{environmentLabel}</span>
          </div>
          <small>{environment.data?.gpu_name ?? environment.data?.gpu ?? (environment.error ? 'GPU 状态不可用' : 'CPU 题不受 GPU 状态影响')}</small>
        </NavLink>
        <div className="sidebar-footer">
          <a className="sidebar-link" href="https://github.com/BabyDrangoner/MyLeetGpu" target="_blank" rel="noreferrer" aria-label="项目仓库（新窗口打开）" title="项目仓库（新窗口打开）">
            <Github size={16} aria-hidden="true" />
            <span className="nav-label">项目仓库</span>
          </a>
        </div>
      </aside>

      <div className="main-content">
        <header className="topbar">
          <span className="topbar-location">{locationLabel}</span>
          <div className="topbar-actions">
            <Link className="execution-target-badge" to="/environment" title="设置 GPU 题执行位置；CPU 题始终在 Worker 本机执行" aria-label={`GPU 执行位置：${executionTargetLabel(execution?.settings?.target)}`}>GPU：{executionTargetLabel(execution?.settings?.target)}</Link>
            <span className="mode-indicator" title={localMode ? `${hostname} · API 本地连接，与 GPU 执行位置无关` : `${hostname} · 认证局域网 API`}>
              {localMode ? '本地 API' : '局域网 API'}
            </span>
            <button className="theme-toggle" type="button" onClick={toggleTheme} aria-label={themeLabel} title={themeLabel}>
              {theme === 'light' ? <Moon size={17} aria-hidden="true" /> : <Sun size={17} aria-hidden="true" />}
            </button>
          </div>
        </header>
        <main className="route-content" id="main-content" tabIndex={-1}>
          <Outlet />
        </main>
      </div>
    </div>
  )
}
