import { Navigate } from 'react-router-dom'

const hasValidToken = () => {
  const token = localStorage.getItem('token')
  if (!token) {
    return false
  }

  try {
    const payloadBase64 = token.split('.')[1]
    const payloadJson = atob(payloadBase64)
    const payload = JSON.parse(payloadJson)

    if (!payload.exp) {
      return true
    }

    const isValid = payload.exp * 1000 > Date.now()
    if (!isValid) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
    }

    return isValid
  } catch {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    return false
  }
}

function ProtectedRoute({ children }) {
  if (!hasValidToken()) {
    return <Navigate to="/login" replace />
  }

  return children
}

export { hasValidToken }
export default ProtectedRoute
