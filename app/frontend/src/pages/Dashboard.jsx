import { useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'

function Dashboard() {
  const navigate = useNavigate()

  const username = useMemo(() => {
    try {
      const rawUser = localStorage.getItem('user')
      if (!rawUser) {
        return 'User'
      }
      const parsed = JSON.parse(rawUser)
      return parsed.username || 'User'
    } catch {
      return 'User'
    }
  }, [])

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    navigate('/login', { replace: true })
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-100 to-white px-4 py-10 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-4xl rounded-2xl bg-white p-8 shadow-sm ring-1 ring-slate-200">
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-3xl font-bold text-slate-900">Dashboard</h1>
          <button
            type="button"
            onClick={handleLogout}
            className="rounded-xl bg-rose-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-500"
          >
            Logout
          </button>
        </div>

        <p className="mb-6 text-lg text-slate-700">Welcome, {username}</p>

        <div className="grid gap-3 sm:grid-cols-2">
          <Link
            to="/enrollment"
            className="rounded-xl bg-blue-600 px-4 py-3 text-center font-semibold text-white transition hover:bg-blue-500"
          >
            Voice Enrollment
          </Link>

          <Link
            to="/verification"
            className="rounded-xl bg-slate-700 px-4 py-3 text-center font-semibold text-white transition hover:bg-slate-600"
          >
            Voice Verification
          </Link>
        </div>
      </div>
    </main>
  )
}

export default Dashboard
