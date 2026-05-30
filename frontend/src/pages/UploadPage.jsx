/**
 * Upload screen — kicks off the whole ingest pipeline on the backend.
 *
 * One upload = parse → classify → hybrid chunk → embed → FAISS (document_ingestion.py).
 * User can force document type or let auto-classify + confidence gap handle it.
 */
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { documentsApi } from '../services/api'

const TYPES = [
  'Employment Contract',
  'Lease Agreement',
  'Rental Agreement',
  'NDA',
  'Vendor Agreement',
  'Service Agreement',
  'General Legal Document',
]

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [language, setLanguage] = useState('en')
  const [docType, setDocType] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const upload = async () => {
    if (!file) return
    setLoading(true)
    try {
      const { data } = await documentsApi.upload(file, language, docType || null)
      setResult(data)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-xl">
      <h2 className="text-2xl font-bold mb-6">Upload Legal Document</h2>

      <input
        type="file"
        accept=".pdf,.docx"
        className="mb-4 block w-full"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <select
        className="w-full mb-3 p-3 rounded bg-slate-800 border border-slate-700"
        value={language}
        onChange={(e) => setLanguage(e.target.value)}
      >
        <option value="en">English</option>
        <option value="hi">Hindi</option>
        <option value="hinglish">Hinglish</option>
      </select>

      <select
        className="w-full mb-6 p-3 rounded bg-slate-800 border border-slate-700"
        value={docType}
        onChange={(e) => setDocType(e.target.value)}
      >
        <option value="">Auto-classify</option>
        {TYPES.map((t) => (
          <option key={t} value={t}>
            {t}
          </option>
        ))}
      </select>

      <button
        onClick={upload}
        disabled={loading || !file}
        className="bg-blue-600 px-6 py-3 rounded-lg disabled:opacity-50"
      >
        {loading ? 'Processing...' : 'Upload & Index'}
      </button>

      {result && (
        <div className="mt-6 p-4 bg-slate-800 rounded-lg">
          <p className="text-green-400">{result.message}</p>
          <p>Type: {result.document_type || 'Needs selection'}</p>
          {/* if gap between top-2 classes was small, user must pick type on Dashboard */}
          {result.manual_selection_required && (
            <p className="text-amber-400">Low confidence — please confirm document type on dashboard.</p>
          )}
          <button
            className="mt-4 text-blue-400 underline"
            onClick={() => navigate(`/documents/${result.id}/chat`)}
          >
            Open Chat
          </button>
        </div>
      )}
    </div>
  )
}
