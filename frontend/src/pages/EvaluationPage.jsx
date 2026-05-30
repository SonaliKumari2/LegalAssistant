/**
 * RAGAS evaluation UI — run test questions against a document and see metrics.
 *
 * Backend runs the real RAG pipeline per question, then scores faithfulness / relevance.
 * Good for answering "how do you know the system works?"
 */
import { useEffect, useState } from 'react'
import { evaluationApi, documentsApi } from '../services/api'

export default function EvaluationPage() {
  const [docs, setDocs] = useState([])
  const [history, setHistory] = useState([])
  const [docId, setDocId] = useState('')
  const [questions, setQuestions] = useState('What is the notice period?\nWhat are payment terms?')

  useEffect(() => {
    documentsApi.list().then((r) => setDocs(r.data))
    evaluationApi.history().then((r) => setHistory(r.data))
  }, [])

  const run = async () => {
    const qs = questions.split('\n').filter(Boolean)
    await evaluationApi.run({ document_id: Number(docId), questions: qs })
    const h = await evaluationApi.history()
    setHistory(h.data)
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">RAG Evaluation Dashboard</h2>
      <div className="grid md:grid-cols-2 gap-8">
        <div className="bg-legal-900 p-6 rounded-xl border border-slate-800">
          <select
            className="w-full mb-4 p-2 bg-slate-800 rounded"
            value={docId}
            onChange={(e) => setDocId(e.target.value)}
          >
            <option value="">Select document</option>
            {docs.map((d) => (
              <option key={d.id} value={d.id}>
                {d.title}
              </option>
            ))}
          </select>
          <textarea
            className="w-full h-32 p-3 bg-slate-800 rounded mb-4"
            value={questions}
            onChange={(e) => setQuestions(e.target.value)}
            placeholder="One question per line"
          />
          <button onClick={run} className="bg-blue-600 px-4 py-2 rounded-lg">
            Run RAGAS Evaluation
          </button>
        </div>
        <div>
          <h3 className="font-medium mb-4">History</h3>
          {history.map((e) => (
            <div key={e.id} className="mb-4 p-4 bg-slate-800 rounded-lg text-sm">
              <p>Faithfulness: {e.faithfulness?.toFixed(2) ?? '—'}</p>
              <p>Answer Relevance: {e.answer_relevance?.toFixed(2) ?? '—'}</p>
              <p>Precision: {e.precision?.toFixed(2) ?? '—'} · Recall: {e.recall?.toFixed(2) ?? '—'}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
