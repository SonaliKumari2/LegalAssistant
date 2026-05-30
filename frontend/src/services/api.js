/**
 * All HTTP calls to our FastAPI backend.
 * Vite proxies /api → localhost:8000 (see vite.config.js).
 *
 * The interceptor attaches JWT from localStorage so every request is authenticated
 * after login — interview talking point for "how we secure the API".
 */
import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

export const authApi = {
  register: (data) => api.post('/auth/register', data),
  login: (data) => api.post('/auth/login', data),
  me: () => api.get('/auth/me'),
}

/** Lists every feature we implemented — good to hit in browser during demo */
export const featuresApi = {
  list: () => api.get('/features/'),
}

export const documentsApi = {
  list: () => api.get('/documents/'),
  /** Shows chunk counts, classification scores — Pipeline page in UI */
  pipeline: (id) => api.get(`/documents/${id}/pipeline`),
  upload: (file, language = 'en', documentType = null) => {
    const form = new FormData()
    form.append('file', file)
    form.append('language', language)
    // if user picked a type upfront, skip auto-classify ambiguity
    if (documentType) form.append('document_type', documentType)
    return api.post('/documents/upload', form)
  },
  get: (id) => api.get(`/documents/${id}`),
  remove: (id) => api.delete(`/documents/${id}`),
  /** When confidence gap was low, user fixes type from Dashboard */
  setType: (id, document_type) => api.patch(`/documents/${id}/type`, { document_type }),
}

/** RAG Q&A — hits rag_pipeline.py on the server */
export const qaApi = {
  ask: (document_id, question, language = 'en') =>
    api.post('/qa/ask', { document_id, question, language }),
}

export const summariesApi = {
  generate: (document_id, language = 'en') =>
    api.post('/summaries/generate', { document_id, language }),
}

export const risksApi = {
  analyze: (document_id, language = 'en') =>
    api.post(`/risks/analyze/${document_id}`, null, { params: { language } }),
  get: (document_id) => api.get(`/risks/${document_id}`),
}

export const evaluationApi = {
  run: (payload) => api.post('/evaluation/run', payload),
  history: () => api.get('/evaluation/history'),
}

export default api
