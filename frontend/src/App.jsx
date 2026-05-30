/**
 * App routing — public pages (login/register) vs protected app shell (Layout).
 *
 * Anything under Layout requires JWT (see PrivateRoute + AuthContext).
 * Document routes use :id from the URL for which contract we're working on.
 */
import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import { useAuth } from './contexts/AuthContext'
import ChatPage from './pages/ChatPage'
import Dashboard from './pages/Dashboard'
import DocumentViewer from './pages/DocumentViewer'
import EvaluationPage from './pages/EvaluationPage'
import Login from './pages/Login'
import Profile from './pages/Profile'
import Register from './pages/Register'
import RiskPage from './pages/RiskPage'
import SummaryPage from './pages/SummaryPage'
import UploadPage from './pages/UploadPage'

function PrivateRoute({ children }) {
  const { user, loading } = useAuth()
  if (loading) return <div className="p-8">Loading...</div>
  return user ? children : <Navigate to="/login" />
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route
        element={
          <PrivateRoute>
            <Layout />
          </PrivateRoute>
        }
      >
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/documents/:id" element={<DocumentViewer />} />
        <Route path="/documents/:id/summary" element={<SummaryPage />} />
        <Route path="/documents/:id/risks" element={<RiskPage />} />
        <Route path="/documents/:id/chat" element={<ChatPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="/profile" element={<Profile />} />
      </Route>
      <Route path="*" element={<Navigate to="/dashboard" />} />
    </Routes>
  )
}
