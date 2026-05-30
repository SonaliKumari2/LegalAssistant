/**
 * Legal Q&A chat — the user-facing side of our RAG pipeline.
 *
 * Each "Send" calls POST /api/qa/ask → backend does FAISS + BGE rerank + Gemini.
 * We show citations (page, section) so answers aren't a black box.
 */
import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { qaApi } from '../services/api'

export default function ChatPage() {
  const { id } = useParams() // document id from URL /documents/:id/chat
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [language, setLanguage] = useState('en')
  const [loading, setLoading] = useState(false)

  const ask = async (e) => {
    e.preventDefault()
    if (!question.trim()) return
    setLoading(true)
    try {
      // backend: retrieve chunks → rerank → Gemini (see rag_pipeline.py)
      const { data } = await qaApi.ask(Number(id), question, language)
      setMessages((m) => [
        ...m,
        { role: 'user', text: question },
        { role: 'assistant', ...data },
      ])
      setQuestion('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto flex flex-col h-[80vh]">
      <h2 className="text-2xl font-bold mb-4">Legal Q&A</h2>

      {/* en / hi / hinglish — server adds cross-lingual instructions for Gemini */}
      <select
        className="mb-4 w-40 p-2 rounded bg-slate-800"
        value={language}
        onChange={(e) => setLanguage(e.target.value)}
      >
        <option value="en">English</option>
        <option value="hi">Hindi</option>
        <option value="hinglish">Hinglish</option>
      </select>

      <div className="flex-1 overflow-y-auto space-y-4 mb-4">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`p-4 rounded-xl ${m.role === 'user' ? 'bg-blue-900/40 ml-12' : 'bg-slate-800 mr-12'}`}
          >
            {m.role === 'user' ? (
              <p>{m.text}</p>
            ) : (
              <>
                <p className="whitespace-pre-wrap">{m.answer}</p>
                {/* confidence comes from reranker score — rough but useful for UI */}
                <p className="text-xs text-slate-400 mt-2">Confidence: {(m.confidence * 100).toFixed(0)}%</p>
                {m.citations?.length > 0 && (
                  <div className="mt-3 text-sm border-t border-slate-700 pt-2">
                    <p className="font-medium text-blue-300">Sources</p>
                    {m.citations.map((c, j) => (
                      <p key={j} className="text-slate-400">
                        p.{c.page ?? '?'} · {c.section || 'Section N/A'}
                      </p>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        ))}
      </div>

      <form onSubmit={ask} className="flex gap-2">
        <input
          className="flex-1 p-3 rounded-lg bg-slate-800 border border-slate-700"
          placeholder="Ask about termination, payment terms..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button disabled={loading} className="bg-blue-600 px-6 rounded-lg">
          Send
        </button>
      </form>
    </div>
  )
}
