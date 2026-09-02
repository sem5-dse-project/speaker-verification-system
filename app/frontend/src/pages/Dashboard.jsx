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
    <PageShell>
      <div className="card mx-auto max-w-4xl p-8">
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <h1 className="heading-1 text-3xl">Dashboard</h1>
          <button
            type="button"
            onClick={handleLogout}
            className="rounded-xl bg-rose-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-500 dark:bg-rose-700 dark:hover:bg-rose-600"
          >
            Logout
          </button>
        </div>

        <p className="mb-6 text-lg text-body">Welcome, {username}</p>

        <div className="grid gap-3 sm:grid-cols-2">
          <Link
            to="/enrollment"
            className="rounded-xl bg-brand-600 px-4 py-3 text-center font-semibold text-white transition hover:bg-brand-500 dark:bg-brand-700 dark:hover:bg-brand-600"
          >
            Voice Enrollment
          </Link>

          <Link
            to="/verification"
            className="rounded-xl bg-brand-800 px-4 py-3 text-center font-semibold text-white transition hover:bg-brand-700 dark:bg-brand-900 dark:hover:bg-brand-800"
          >
            Voice Verification
          </Link>
        </div>
      </div>
    </PageShell>
  )
}

export default Dashboard
