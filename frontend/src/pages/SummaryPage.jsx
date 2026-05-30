/**
 * Contract summary page — calls summarizer.py (structured JSON from Gemini).
 *
 * Not the same as RAG chat: this reads the file and summarizes in one shot.
 * Language toggle re-fetches in English or Hindi.
 */
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { summariesApi } from '../services/api'

export default function SummaryPage() {
  const { id } = useParams()
  const [summary, setSummary] = useState(null)
  const [language, setLanguage] = useState('en')

  const load = () => {
    summariesApi.generate(Number(id), language).then((r) => setSummary(r.data))
  }

  useEffect(() => {
    load()
  }, [id, language])

  if (!summary) return <p>Generating summary...</p>

  const sections = [
    ['Executive Summary', summary.executive_summary],
    ['Payment Terms', summary.payment_terms],
    ['Termination', summary.termination_conditions],
  ]

  return (
    <div className="max-w-3xl">
      <div className="flex gap-4 mb-6">
        <h2 className="text-2xl font-bold flex-1">Contract Summary</h2>
        <select value={language} onChange={(e) => setLanguage(e.target.value)} className="bg-slate-800 p-2 rounded">
          <option value="en">English</option>
          <option value="hi">Hindi</option>
        </select>
      </div>
      {sections.map(([title, body]) => (
        <section key={title} className="mb-6 bg-legal-900 p-5 rounded-xl border border-slate-800">
          <h3 className="text-blue-400 font-medium mb-2">{title}</h3>
          <p className="text-slate-300 whitespace-pre-wrap">{body}</p>
        </section>
      ))}
      <ListSection title="Key Obligations" items={summary.key_obligations} />
      <ListSection title="Important Dates" items={summary.important_dates} />
      <ListSection title="Risks" items={summary.risks} />
      <ListSection title="Legal Concerns" items={summary.legal_concerns} />
    </div>
  )
}

function ListSection({ title, items }) {
  if (!items?.length) return null
  return (
    <section className="mb-6 bg-legal-900 p-5 rounded-xl border border-slate-800">
      <h3 className="text-blue-400 font-medium mb-2">{title}</h3>
      <ul className="list-disc pl-5 text-slate-300 space-y-1">
        {items.map((x, i) => (
          <li key={i}>{x}</li>
        ))}
      </ul>
    </section>
  )
}
