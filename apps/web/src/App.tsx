import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { StatusView } from './components/StatusView'
import { ToastProvider } from './components/Toast'
import { ThemeProvider } from './hooks/useTheme'
import { EnvironmentPage } from './pages/EnvironmentPage'
import { ProblemListPage } from './pages/ProblemListPage'

const WorkspacePage = lazy(() => import('./pages/WorkspacePage').then((module) => ({ default: module.WorkspacePage })))
const VersionsPage = lazy(() => import('./pages/VersionsPage').then((module) => ({ default: module.VersionsPage })))

export function App() {
  return (
    <ThemeProvider>
      <ToastProvider>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<Navigate replace to="/problems" />} />
            <Route path="/problems" element={<ProblemListPage />} />
            <Route path="/problems/:slug" element={
              <Suspense fallback={<div className="page"><StatusView kind="loading" title="正在打开编程工作台" description="正在加载代码编辑器…" /></div>}>
                <WorkspacePage />
              </Suspense>
            } />
            <Route path="/problems/:slug/versions" element={
              <Suspense fallback={<div className="page"><StatusView kind="loading" title="正在打开性能版本" description="正在加载版本与代码比较视图…" /></div>}>
                <VersionsPage />
              </Suspense>
            } />
            <Route path="/environment" element={<EnvironmentPage />} />
            <Route path="*" element={<Navigate replace to="/problems" />} />
          </Route>
        </Routes>
      </ToastProvider>
    </ThemeProvider>
  )
}
