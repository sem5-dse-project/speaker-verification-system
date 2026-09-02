import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  CheckCircle2,
  Clock3,
  Mic,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Volume2,
} from 'lucide-react'
import PageShell from '../components/PageShell.jsx'
import FeatureTile from '../components/FeatureTile.jsx'
import ProgressRing from '../components/ProgressRing.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import api from '../services/api.js'

const REQUIRED_SAMPLES = 3

const DECISION_LABELS = {
  ACCEPT: { label: 'Verified', className: 'text-emerald-600 dark:text-emerald-400' },
  REJECT: { label: 'Rejected', className: 'text-rose-600 dark:text-rose-400' },
  REPLAY: { label: 'Replay blocked', className: 'text-rose-600 dark:text-rose-400' },
  SYNTHETIC: { label: 'Synthetic blocked', className: 'text-rose-600 dark:text-rose-400' },
  UNCERTAIN: { label: 'Uncertain', className: 'text-amber-600 dark:text-amber-400' },
  NO_SPEECH: { label: 'No speech', className: 'text-amber-600 dark:text-amber-400' },
}

function formatWhen(value) {
  if (!value) {
    return 'Unknown'
  }
  try {
    return new Date(value).toLocaleString()
  } catch {
    return String(value)
  }
}

function Dashboard() {
  const { user } = useAuth()
  const [loading, setLoading] = useState(true)
  const [template, setTemplate] = useState(null)
  const [verificationLogs, setVerificationLogs] = useState([])
  const [enrollmentCount, setEnrollmentCount] = useState(0)

  const username = useMemo(() => user?.username || 'User', [user])

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      setLoading(true)
      try {
        const response = await api.get('/voice/history')
        if (cancelled) {
          return
        }

        const history = response.data?.history || []
        setTemplate(response.data?.template || null)
        setVerificationLogs((response.data?.verification_logs || []).slice(0, 5))
        setEnrollmentCount(
          history.filter((row) => row.sample_type === 'enrollment').length,
        )
      } catch {
        if (!cancelled) {
          setTemplate(null)
          setVerificationLogs([])
          setEnrollmentCount(0)
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [])

  const isEnrolled = Boolean(template?.has_embedding)
  const enrollmentProgress = isEnrolled
    ? REQUIRED_SAMPLES
    : Math.min(enrollmentCount, REQUIRED_SAMPLES)
  const lastLog = verificationLogs[0]

  return (
    <PageShell narrow showNav>
      <div className="space-y-5 sm:space-y-6">
        <header className="space-y-1">
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-brand-700 dark:text-brand-400">
            Voice identity hub
          </p>
          <h1 className="heading-1">Welcome back, {username}</h1>
          <p className="text-sm text-muted sm:text-base">
            Enroll your voice once, then verify securely with replay protection.
          </p>
        </header>

        <section className="card grid gap-5 sm:grid-cols-[auto_1fr] sm:items-center">
          <ProgressRing
            value={enrollmentProgress}
            max={REQUIRED_SAMPLES}
            label="Enrollment progress"
          />
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              {isEnrolled ? (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-3 py-1 text-xs font-bold uppercase tracking-wide text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-200">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Voice enrolled
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-100 px-3 py-1 text-xs font-bold uppercase tracking-wide text-amber-800 dark:bg-amber-950/50 dark:text-amber-200">
                  <ShieldAlert className="h-3.5 w-3.5" />
                  Enrollment incomplete
                </span>
              )}
              {!loading && template?.updated_at && (
                <span className="inline-flex items-center gap-1 text-xs text-subtle">
                  <Clock3 className="h-3.5 w-3.5" />
                  Updated {formatWhen(template.updated_at)}
                </span>
              )}
            </div>
            <p className="text-sm text-body">
              {isEnrolled
                ? 'Your voice template is ready. Run a verification check or re-enroll if your voice or microphone setup changed.'
                : `Complete ${REQUIRED_SAMPLES} enrollment recordings to build your secure voice template.`}
            </p>
            {!isEnrolled && (
              <Link to="/enrollment" className="btn-primary-lg sm:max-w-xs">
                Continue enrollment
              </Link>
            )}
          </div>
        </section>

        <section className="grid gap-3 sm:grid-cols-2">
          <FeatureTile
            to="/enrollment"
            icon={Mic}
            title="Voice Enrollment"
            description="Record 3 samples · build your voice template"
            variant="primary"
          />
          <FeatureTile
            to="/verification"
            icon={ShieldCheck}
            title="Voice Verification"
            description="Live check with replay & speaker match"
            variant="muted"
          />
          <FeatureTile
            to="/voice-login"
            icon={Sparkles}
            title="Login with Voice"
            description="Identify by voice, confirm with password"
            variant="muted"
          />
        </section>

        <section className="tips-strip">
          <div className="tip-chip">
            <Volume2 className="mb-1 h-4 w-4" />
            Use a quiet room with minimal background noise.
          </div>
          <div className="tip-chip">
            <Mic className="mb-1 h-4 w-4" />
            Hold the mic 15–30 cm away and speak naturally.
          </div>
          <div className="tip-chip">
            <ShieldCheck className="mb-1 h-4 w-4" />
            Replay from a phone speaker is blocked by design.
          </div>
        </section>

        <section className="card">
          <div className="mb-4 flex items-center justify-between gap-3">
            <h2 className="heading-2">Recent verifications</h2>
            {!loading && verificationLogs.length > 0 && (
              <Link to="/verification" className="link-primary text-sm">
                Verify again
              </Link>
            )}
          </div>

          {loading && <p className="text-sm text-subtle">Loading activity...</p>}

          {!loading && verificationLogs.length === 0 && (
            <div className="rounded-xl border border-dashed border-brand-200 bg-brand-50/40 px-4 py-6 text-center dark:border-brand-900/30 dark:bg-brand-950/15">
              <p className="text-sm font-semibold text-body">No verification attempts yet</p>
              <p className="mt-1 text-sm text-muted">
                Complete enrollment first, then run your first voice check.
              </p>
            </div>
          )}

          {!loading && verificationLogs.length > 0 && (
            <ul className="space-y-2">
              {verificationLogs.map((log) => {
                const decisionMeta = DECISION_LABELS[log.decision] || {
                  label: log.decision || 'Unknown',
                  className: 'text-muted',
                }

                return (
                  <li key={log.id} className="list-item flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <p className={`text-sm font-semibold ${decisionMeta.className}`}>
                        {decisionMeta.label}
                      </p>
                      <p className="text-xs text-subtle">{formatWhen(log.created_at)}</p>
                    </div>
                    {typeof log.score === 'number' && (
                      <p className="text-sm font-bold tabular-nums text-body">
                        {log.score.toFixed(3)}
                      </p>
                    )}
                  </li>
                )
              })}
            </ul>
          )}

          {!loading && lastLog && (
            <p className="mt-3 text-xs text-subtle">
              Last result: {DECISION_LABELS[lastLog.decision]?.label || lastLog.decision}
              {typeof lastLog.score === 'number' ? ` · score ${lastLog.score.toFixed(3)}` : ''}
            </p>
          )}
        </section>
      </div>
    </PageShell>
  )
}

export default Dashboard
