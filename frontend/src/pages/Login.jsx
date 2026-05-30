import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

export default function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    try {
      await login(email, password)
      navigate('/dashboard')
    } catch {
      setError('Invalid credentials')
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-950">
      <form onSubmit={submit} className="w-full max-w-md bg-legal-900 p-8 rounded-2xl border border-slate-800">
        <h1 className="text-2xl font-bold text-blue-400 mb-2">Kanooni Sahayak</h1>
        <p className="text-slate-400 mb-6">AI Legal Document Assistant</p>
        {error && <p className="text-red-400 mb-4">{error}</p>}
        <input
          className="w-full mb-3 p-3 rounded bg-slate-800 border border-slate-700"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          className="w-full mb-6 p-3 rounded bg-slate-800 border border-slate-700"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button className="w-full bg-blue-600 hover:bg-blue-500 py-3 rounded-lg font-medium">
          Login
        </button>
        <p className="mt-4 text-center text-slate-400">
          No account? <Link to="/register" className="text-blue-400">Register</Link>
        </p>
      </form>
    </div>
  )
}
