import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { ToastProvider } from './components/Toast'
import { EnvironmentPage } from './pages/EnvironmentPage'
import { ProblemListPage } from './pages/ProblemListPage'
import { VersionsPage } from './pages/VersionsPage'
import { WorkspacePage } from './pages/WorkspacePage'

export function App() {
  return (
    <ToastProvider>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Navigate replace to="/problems" />} />
          <Route path="/problems" element={<ProblemListPage />} />
          <Route path="/problems/:slug" element={<WorkspacePage />} />
          <Route path="/problems/:slug/versions" element={<VersionsPage />} />
          <Route path="/environment" element={<EnvironmentPage />} />
          <Route path="*" element={<Navigate replace to="/problems" />} />
        </Route>
      </Routes>
    </ToastProvider>
  )
}
