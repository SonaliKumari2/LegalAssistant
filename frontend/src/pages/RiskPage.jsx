/**
 * Risky clause UI — triggers framework + open-ended analysis on the backend.
 *
 * Shows High / Medium / Low counts. Each card is one flagged clause with optional page cite.
 */
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { risksApi } from '../services/api'

const severityColor = { High: 'text-red-400 border-red-800', Medium: 'text-amber-400 border-amber-800', Low: 'text-green-400 border-green-800' }

export default function RiskPage() {
  const { id } = useParams()
  const [data, setData] = useState(null)

  useEffect(() => {
    // POST analyze runs risk_analyzer.py (JSON framework + open-ended pass)
    risksApi.analyze(Number(id)).then((r) => setData(r.data))
  }, [id])

  if (!data) return <p>Analyzing risks...</p>

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Risk Analysis</h2>
      <div className="flex gap-4 mb-8">
        <Stat label="High" count={data.high_count} color="red" />
        <Stat label="Medium" count={data.medium_count} color="amber" />
        <Stat label="Low" count={data.low_count} color="green" />
      </div>
      <div className="space-y-4">
        {data.risks.map((r, i) => (
          <div key={i} className={`p-5 rounded-xl border bg-legal-900 ${severityColor[r.severity]}`}>
            <div className="flex justify-between mb-2">
              <span className="font-semibold">{r.risk_type || 'Risk'}</span>
              <span>{r.severity}</span>
            </div>
            <p className="text-slate-300 text-sm mb-2">{r.clause}</p>
            <p className="text-slate-400">{r.explanation}</p>
            {(r.page || r.section) && (
              <p className="text-xs mt-2 text-slate-500">
                Citation: p.{r.page} · {r.section}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function Stat({ label, count, color }) {
  return (
    <div className={`px-6 py-3 rounded-lg bg-${color}-900/20 border border-${color}-800`}>
      <p className="text-2xl font-bold">{count}</p>
      <p className="text-sm text-slate-400">{label}</p>
    </div>
  )
}
