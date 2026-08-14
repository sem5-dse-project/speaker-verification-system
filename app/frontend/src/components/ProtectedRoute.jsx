import { Navigate } from 'react-router-dom'
import { useEffect } from 'react'
import { useAuth } from '../context/AuthContext.jsx'
import { isTokenValid } from '../utils/jwt.js'

const hasValidToken = (tokenFromArg) => {
  const token =
    tokenFromArg || localStorage.getItem('token') || sessionStorage.getItem('token')

  if (!token) {
    return false
  }

  const valid = isTokenValid(token)
  if (!valid) {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    sessionStorage.removeItem('token')
    sessionStorage.removeItem('user')
  }

  return valid
}

function ProtectedRoute({ children }) {
  const { token, logout } = useAuth()
  const valid = hasValidToken(token)

  useEffect(() => {
    if (token && !valid) {
      logout()
    }
  }, [token, valid, logout])

  if (!valid) {
    return <Navigate to="/login" replace />
  }

  return children
}

export { hasValidToken }
export default ProtectedRoute
