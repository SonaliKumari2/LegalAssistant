import { useAuth } from '../contexts/AuthContext'

export default function Profile() {
  const { user } = useAuth()
  return (
    <div className="max-w-md">
      <h2 className="text-2xl font-bold mb-6">Profile</h2>
      <div className="bg-legal-900 p-6 rounded-xl border border-slate-800 space-y-2">
        <p>
          <span className="text-slate-400">Name:</span> {user?.full_name || '—'}
        </p>
        <p>
          <span className="text-slate-400">Email:</span> {user?.email}
        </p>
        <p>
          <span className="text-slate-400">Role:</span> {user?.role}
        </p>
      </div>
    </div>
  )
}
