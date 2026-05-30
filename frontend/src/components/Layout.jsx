import { Link, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

const nav = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/upload', label: 'Upload' },
  { to: '/evaluation', label: 'Evaluation' },
  { to: '/profile', label: 'Profile' },
]

export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  return (
    <div className="min-h-screen flex">
      <aside className="w-64 bg-legal-900 border-r border-slate-800 p-6 flex flex-col">
        <h1 className="text-xl font-bold text-blue-400 mb-8">Kanooni Sahayak</h1>
        <nav className="flex flex-col gap-2 flex-1">
          {nav.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              className="px-3 py-2 rounded-lg hover:bg-slate-800 text-slate-300 hover:text-white"
            >
              {item.label}
            </Link>
          ))}
        </nav>
        <div className="text-sm text-slate-400">
          <p>{user?.email}</p>
          <button
            onClick={() => {
              logout()
              navigate('/login')
            }}
            className="mt-2 text-red-400 hover:underline"
          >
            Logout
          </button>
        </div>
      </aside>
      <main className="flex-1 p-8 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
