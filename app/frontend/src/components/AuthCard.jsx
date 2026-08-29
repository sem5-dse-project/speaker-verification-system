import { ShieldCheck, Waves } from 'lucide-react'

function AuthCard({ title, subtitle, children }) {
  return (
    <section className="grid w-full max-w-5xl overflow-hidden rounded-2xl bg-white shadow-xl shadow-slate-200/70 ring-1 ring-slate-200 dark:bg-slate-900 dark:shadow-slate-950/40 dark:ring-slate-800 lg:grid-cols-2">
      <aside className="relative hidden overflow-hidden bg-gradient-to-br from-teal-700 via-teal-600 to-emerald-500 p-10 text-white lg:block">
        <div className="absolute -left-12 -top-12 h-48 w-48 rounded-full bg-white/10 blur-lg" />
        <div className="absolute -bottom-16 right-0 h-56 w-56 rounded-full bg-emerald-300/20 blur-2xl" />

        <div className="relative z-10 flex h-full flex-col justify-between">
          <div className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-white/15 ring-1 ring-white/40 backdrop-blur">
            <ShieldCheck className="h-6 w-6" />
          </div>

          <div className="space-y-4">
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-teal-100">
              Voice Authentication
            </p>
            <h2 className="text-3xl font-extrabold leading-tight">
              Secure identity with your unique voiceprint
            </h2>
            <p className="max-w-md text-sm text-teal-100/90">
              Register once, then enroll and verify your voice with modern biometric security controls.
            </p>
            <div className="inline-flex items-center gap-2 rounded-lg bg-white/15 px-3 py-2 text-xs font-medium backdrop-blur">
              <Waves className="h-4 w-4" />
              Trusted secure onboarding
            </div>
          </div>
        </div>
      </aside>

      <div className="p-6 sm:p-8 lg:p-10">
        <header className="mb-6 space-y-2">
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-slate-100">{title}</h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 sm:text-base">{subtitle}</p>
        </header>

        {children}
      </div>
    </section>
  )
}

export default AuthCard