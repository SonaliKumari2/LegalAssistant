import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

export default function Register() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ email: '', password: '', full_name: '' })
  const [error, setError] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    try {
      await register(form.email, form.password, form.full_name)
      navigate('/dashboard')
    } catch {
      setError('Registration failed')
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-950">
      <form onSubmit={submit} className="w-full max-w-md bg-legal-900 p-8 rounded-2xl border border-slate-800">
        <h1 className="text-2xl font-bold mb-6">Create Account</h1>
        {error && <p className="text-red-400 mb-4">{error}</p>}
        {['full_name', 'email', 'password'].map((field) => (
          <input
            key={field}
            type={field === 'password' ? 'password' : 'text'}
            className="w-full mb-3 p-3 rounded bg-slate-800 border border-slate-700"
            placeholder={field.replace('_', ' ')}
            value={form[field]}
            onChange={(e) => setForm({ ...form, [field]: e.target.value })}
          />
        ))}
        <button className="w-full bg-blue-600 py-3 rounded-lg">Register</button>
        <p className="mt-4 text-center text-slate-400">
          <Link to="/login" className="text-blue-400">Back to login</Link>
        </p>
      </form>
    </div>
  )
}
