import { Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'

function AdminRoute({ children }) {
  const { isAuthenticated, isAdmin } = useAuth()

  if (!isAuthenticated) {
    return <Navigate to="/admin/login" replace />
  }

  if (!isAdmin) {
    return <Navigate to="/dashboard" replace />
  }

  return children
}

export default AdminRoute
