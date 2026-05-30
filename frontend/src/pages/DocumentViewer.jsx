/**
 * "Pipeline" page — interview gold.
 *
 * Calls GET /documents/:id/pipeline and shows checkmarks for each step
 * (FAISS, hybrid chunks, BGE, classification scores). Open this during demos.
 */
import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { documentsApi } from '../services/api'

export default function DocumentViewer() {
  const { id } = useParams()
  const [doc, setDoc] = useState(null)
  const [pipeline, setPipeline] = useState(null)

  useEffect(() => {
    documentsApi.get(Number(id)).then((r) => setDoc(r.data))
    documentsApi.pipeline(Number(id)).then((r) => setPipeline(r.data))
  }, [id])

  if (!doc) return <p>Loading...</p>

  const check = pipeline?.interview_checklist || {}
  const pipe = pipeline?.pipeline || {}

  return (
    <div className="max-w-3xl">
      <h2 className="text-2xl font-bold mb-2">{doc.title}</h2>
      <p className="text-slate-400 mb-6">
        {doc.document_type || 'Pending classification'} · {doc.page_count} pages
      </p>

      <section className="mb-8 bg-legal-900 border border-slate-800 rounded-xl p-5">
        <h3 className="text-blue-400 font-medium mb-3">Pipeline (for interview demo)</h3>
        <ul className="text-sm space-y-1 text-slate-300">
          {Object.entries(check).map(([k, v]) => (
            <li key={k}>
              {v ? '✅' : '❌'} {k.replace(/_/g, ' ')}
            </li>
          ))}
        </ul>
        {pipe.chunking && (
          <div className="mt-4 text-xs text-slate-400">
            <p>Chunking: {pipe.chunking.strategy}</p>
            <p>
              Small: {pipe.chunking.small_chunk_count} ({pipe.chunking.small_token_range?.join('-')} tok) · Large:{' '}
              {pipe.chunking.large_chunk_count} ({pipe.chunking.large_token_range?.join('-')} tok)
            </p>
          </div>
        )}
        {pipeline?.classification_scores && (
          <div className="mt-4">
            <p className="text-xs text-slate-500 mb-1">Classification scores:</p>
            <pre className="text-xs bg-slate-950 p-2 rounded overflow-auto">
              {JSON.stringify(pipeline.classification_scores, null, 2)}
            </pre>
          </div>
        )}
      </section>

      <div className="flex gap-4">
        <Link to={`/documents/${id}/summary`} className="bg-blue-600 px-4 py-2 rounded-lg">
          Summary
        </Link>
        <Link to={`/documents/${id}/risks`} className="bg-orange-600 px-4 py-2 rounded-lg">
          Risks
        </Link>
        <Link to={`/documents/${id}/chat`} className="bg-green-600 px-4 py-2 rounded-lg">
          Chat
        </Link>
      </div>
    </div>
  )
}
