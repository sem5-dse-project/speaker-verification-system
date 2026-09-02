import { Link } from 'react-router-dom'
import { ShieldCheck, Waves } from 'lucide-react'

const CURRENT_YEAR = new Date().getFullYear()

function AppFooter({ variant = 'app', withBottomNav = false }) {
  const footerClass = [
    'app-footer',
    variant === 'auth' ? 'app-footer-auth' : '',
    variant === 'admin' ? 'app-footer-admin' : '',
    withBottomNav ? 'app-footer-with-nav' : '',
  ]
    .filter(Boolean)
    .join(' ')

  if (variant === 'auth' || variant === 'admin') {
    return (
      <footer className={footerClass}>
        <div className="app-footer-inner">
          <div className="flex items-center gap-2.5">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-[#004d40] to-[#00a884] text-white">
              <Waves className="h-4 w-4" />
            </span>
            <div>
              <p className="text-sm font-extrabold text-slate-900 dark:text-slate-100">VoiceAuth</p>
              <p className="text-xs text-slate-600 dark:text-slate-400">
                Biometric voice verification · Replay protected
              </p>
            </div>
          </div>

          <p className="max-w-xl text-sm leading-relaxed text-slate-600 dark:text-slate-400">
            Secure speaker enrollment and verification using ECAPA-TDNN embeddings, inverted-Mel replay
            detection, and optional LA spoof screening for research-grade voice authentication.
          </p>

          <nav className="flex flex-wrap gap-x-4 gap-y-2 text-sm font-semibold">
            <Link to="/login" className="link-primary">
              Sign in
            </Link>
            <Link to="/register" className="link-primary">
              Register
            </Link>
            <Link to="/voice-login" className="link-primary">
              Voice login
            </Link>
            <Link to="/admin/login" className="link-primary">
              Admin
            </Link>
          </nav>

          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
            <ShieldCheck className="h-3.5 w-3.5 text-brand-600 dark:text-brand-400" />
            <span>Research prototype · Local audio stays on your machine</span>
          </div>

          <p className="text-xs text-slate-500 dark:text-slate-500">
            © {CURRENT_YEAR} Speaker Verification System
          </p>
        </div>
      </footer>
    )
  }

  return (
    <footer className={footerClass}>
      <div className="app-footer-inner app-footer-inner-compact">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-[#004d40] to-[#00a884] text-white">
              <Waves className="h-3.5 w-3.5" />
            </span>
            <span className="text-sm font-bold text-slate-900 dark:text-slate-100">VoiceAuth</span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            © {CURRENT_YEAR} · Replay-protected speaker verification
          </p>
        </div>
        <nav className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs font-semibold">
          <Link to="/dashboard" className="link-primary">
            Dashboard
          </Link>
          <Link to="/enrollment" className="link-primary">
            Enroll
          </Link>
          <Link to="/verification" className="link-primary">
            Verify
          </Link>
          <Link to="/voice-login" className="link-primary">
            Voice login
          </Link>
        </nav>
      </div>
    </footer>
  )
}

export default AppFooter
