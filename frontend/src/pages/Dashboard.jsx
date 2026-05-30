/**
 * Home screen — lists all contracts for the logged-in user.
 *
 * If classification confidence gap was too small, we show a dropdown here
 * so the user can pick Employment / NDA / etc. (maps to confidence gap strategy).
 */
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { documentsApi } from '../services/api'

const DOC_TYPES = [
  'Employment Contract',
  'Lease Agreement',
  'Rental Agreement',
  'NDA',
  'Vendor Agreement',
  'Service Agreement',
  'General Legal Document',
]

export default function Dashboard() {
  const [docs, setDocs] = useState([])

  const refresh = () => documentsApi.list().then((r) => setDocs(r.data))

  useEffect(() => {
    refresh()
  }, [])

  const setType = async (id, document_type) => {
    await documentsApi.setType(id, document_type)
    refresh()
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <div>
          <h2 className="text-2xl font-bold">Your Documents</h2>
          <p className="text-slate-400 text-sm mt-1">
            Kanooni Sahayak — RAG pipeline with classification, hybrid chunks, FAISS, BGE rerank
          </p>
        </div>
        <Link to="/upload" className="bg-blue-600 px-4 py-2 rounded-lg">
          Upload Contract
        </Link>
      </div>
      <div className="grid gap-4">
        {docs.map((d) => (
          <div key={d.id} className="bg-legal-900 border border-slate-800 rounded-xl p-5">
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <h3 className="font-semibold">{d.title}</h3>
                <p className="text-slate-400 text-sm">
                  {d.document_type || 'Type pending'} · {d.page_count} pages
                </p>
                {d.manual_selection_required && (
                  <div className="mt-3 p-3 bg-amber-900/30 border border-amber-700 rounded-lg">
                    <p className="text-amber-300 text-sm mb-2">
                      Confidence gap below threshold — select document type manually:
                    </p>
                    <select
                      className="bg-slate-800 p-2 rounded text-sm w-full max-w-md"
                      defaultValue=""
                      onChange={(e) => e.target.value && setType(d.id, e.target.value)}
                    >
                      <option value="">Choose type…</option>
                      {DOC_TYPES.map((t) => (
                        <option key={t} value={t}>
                          {t}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
              <div className="flex flex-col gap-2 text-sm items-end">
                <Link to={`/documents/${d.id}`} className="text-slate-300">
                  Pipeline
                </Link>
                <Link to={`/documents/${d.id}/summary`} className="text-blue-400">
                  Summary
                </Link>
                <Link to={`/documents/${d.id}/risks`} className="text-orange-400">
                  Risks
                </Link>
                <Link to={`/documents/${d.id}/chat`} className="text-green-400">
                  Chat
                </Link>
              </div>
            </div>
          </div>
        ))}
        {!docs.length && <p className="text-slate-500">No documents yet. Upload your first contract.</p>}
      </div>
    </div>
  )
}
