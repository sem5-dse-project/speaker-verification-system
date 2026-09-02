import { useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import PageShell from '../components/PageShell.jsx'
import { useAuth } from '../context/AuthContext.jsx'

function Dashboard() {
  const navigate = useNavigate()
  const { user, logout } = useAuth()

  const username = useMemo(() => {
    return user?.username || 'User'
  }, [user])

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <PageShell narrow>
      <div className="card sm:p-8">
        <div className="mb-5 flex flex-col gap-3 sm:mb-6 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
          <h1 className="heading-1">Dashboard</h1>
          <button
            type="button"
            onClick={handleLogout}
            className="inline-flex min-h-11 w-full items-center justify-center rounded-xl bg-rose-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-rose-500 sm:w-auto dark:bg-rose-700 dark:hover:bg-rose-600"
          >
            Logout
          </button>
        </div>

        <p className="mb-5 text-base text-body sm:mb-6 sm:text-lg">Welcome, {username}</p>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Link to="/enrollment" className="action-link justify-center">
            Voice Enrollment
          </Link>

          <Link to="/verification" className="action-link-muted justify-center">
            Voice Verification
          </Link>
        </div>
      </div>
    </PageShell>
  )
}

export default Dashboard
