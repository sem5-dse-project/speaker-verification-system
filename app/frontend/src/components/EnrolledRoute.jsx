import { useEffect, useState } from 'react'
import { Navigate } from 'react-router-dom'
import api from '../services/api.js'

function EnrolledRoute({ children }) {
  const [status, setStatus] = useState('loading')

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        const response = await api.get('/voice/history')
        if (cancelled) {
          return
        }
        setStatus(response.data?.template?.has_embedding ? 'enrolled' : 'missing')
      } catch {
        if (!cancelled) {
          setStatus('missing')
        }
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [])

  if (status === 'loading') {
    return <p className="px-4 py-10 text-center text-sm text-muted">Checking enrollment...</p>
  }

  if (status === 'missing') {
    return <Navigate to="/enrollment" replace />
  }

  return children
}

export default EnrolledRoute
