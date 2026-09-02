import { NavLink, useNavigate } from 'react-router-dom'
import { LayoutDashboard, Mic, ShieldCheck, LogOut, Waves } from 'lucide-react'
import { useAuth } from '../context/AuthContext.jsx'
import ThemeToggle from './ThemeToggle.jsx'

const NAV_ITEMS = [
  { to: '/dashboard', label: 'Home', icon: LayoutDashboard },
  { to: '/enrollment', label: 'Enroll', icon: Mic },
  { to: '/verification', label: 'Verify', icon: ShieldCheck },
]

function navLinkClass({ isActive }) {
  return isActive ? 'app-nav-link app-nav-link-active' : 'app-nav-link'
}

function AppNav() {
  const navigate = useNavigate()
  const { user, logout } = useAuth()

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <>
      <header className="app-nav-top hidden md:block">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <NavLink to="/dashboard" className="flex items-center gap-2.5">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-[#004d40] to-[#00a884] text-white shadow-md shadow-brand-200/40 dark:shadow-brand-950/40">
              <Waves className="h-4 w-4" />
            </span>
            <span className="text-base font-extrabold tracking-tight text-slate-900 dark:text-slate-100">
              VoiceAuth
            </span>
          </NavLink>

          <nav className="flex items-center gap-1">
            {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} className={navLinkClass}>
                <Icon className="h-4 w-4" />
                {label}
              </NavLink>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <span className="hidden text-sm font-medium text-muted lg:inline">{user?.username}</span>
            <ThemeToggle />
            <button type="button" onClick={handleLogout} className="btn-secondary px-3 py-2">
              <LogOut className="h-4 w-4" />
              Logout
            </button>
          </div>
        </div>
      </header>

      <nav className="app-nav-bottom md:hidden">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} className={navLinkClass}>
            <Icon className="h-5 w-5" />
            <span className="text-[11px] font-semibold">{label}</span>
          </NavLink>
        ))}
      </nav>
    </>
  )
}

export default AppNav
