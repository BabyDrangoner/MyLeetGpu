import { Activity, BookOpenText, Cpu, Github, ShieldCheck } from 'lucide-react'
import { NavLink, Outlet } from 'react-router-dom'
import { api } from '../api/client'
import { useAsync } from '../hooks/useAsync'
import { readableStatus } from '../lib/format'

export function AppShell() {
  const environment = useAsync(() => api.environment(), [])
  const healthy = environment.data?.healthy ?? environment.data?.status === 'healthy'
  const hostname = window.location.hostname
  const localMode = hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1'

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <div className="brand-name">MyLeetGpu</div>
            <div className="brand-tagline">CUDA 本地实验台</div>
          </div>
        </div>

        <nav className="main-nav" aria-label="主导航">
          <NavLink to="/problems" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
            <BookOpenText size={18} />
            <span>题目工作台</span>
          </NavLink>
          <NavLink to="/environment" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
            <Cpu size={18} />
            <span>运行环境</span>
          </NavLink>
        </nav>

        <div className="sidebar-spacer" />
        <NavLink to="/environment" className="environment-mini">
          <div className="environment-mini-head">
            <span className={`health-dot ${environment.loading ? 'checking' : healthy ? 'healthy' : 'unhealthy'}`} />
            <span>{environment.loading ? '正在探测环境' : healthy ? 'GPU 就绪' : '环境需处理'}</span>
          </div>
          <strong>{environment.data?.gpu_name ?? environment.data?.gpu ?? (environment.error ? '无法连接 API' : '等待设备信息')}</strong>
          <small>
            {environment.data
              ? `${environment.data.cuda_runtime_version ?? environment.data.cuda_version ?? 'CUDA 未知'} · ${environment.data.compute_capability ? `sm_${environment.data.compute_capability.replace('.', '')}` : readableStatus(environment.data.status)}`
              : '点击查看诊断'}
          </small>
        </NavLink>
        <div className="trust-notice">
          <ShieldCheck size={16} />
          <span>仅供可信操作者使用</span>
        </div>
        <a className="sidebar-link" href="https://github.com/BabyDrangoner/MyLeetGpu" target="_blank" rel="noreferrer">
          <Github size={15} />
          项目仓库
        </a>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="local-pill">
            <Activity size={14} /> {localMode ? '127.0.0.1 · 本地模式' : `${hostname} · 认证局域网`}
          </div>
          <div className="topbar-note">编译与 GPU 任务串行、安全隔离执行</div>
        </header>
        <div className="route-content">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
