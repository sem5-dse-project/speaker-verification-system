import BrandIcon from './BrandIcon.jsx'
import BrandMark from './BrandMark.jsx'

function AuthCard({ title, subtitle, children }) {
  return (
    <section className="grid w-full overflow-hidden rounded-2xl bg-white shadow-xl shadow-brand-200/50 ring-1 ring-brand-100 dark:bg-surface-900 dark:shadow-black/30 dark:ring-brand-900/25 lg:grid-cols-2">
      <div className="relative overflow-hidden bg-gradient-to-br from-[#004d40] via-teal-800 to-[#00a884] px-5 py-5 text-white lg:hidden">
        <div className="absolute -right-8 -top-8 h-32 w-32 rounded-full bg-white/10 blur-lg" />
        <div className="relative z-10">
          <BrandMark
            size="md"
            variant="on-dark"
            subtitle="Secure identity with your unique voiceprint"
          />
        </div>
      </div>

      <aside className="relative hidden overflow-hidden bg-gradient-to-br from-[#004d40] via-teal-800 to-[#00a884] p-10 text-white lg:block">
        <div className="absolute -left-12 -top-12 h-48 w-48 rounded-full bg-white/10 blur-lg" />
        <div className="absolute -bottom-16 right-0 h-56 w-56 rounded-full bg-emerald-300/20 blur-2xl" />

        <div className="relative z-10 flex h-full flex-col justify-between">
          <BrandIcon size="lg" className="drop-shadow-md" />

          <div className="space-y-4">
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-emerald-100/90">
              Voice Authentication
            </p>
            <h2 className="text-3xl font-extrabold leading-tight">
              Secure identity with your unique voiceprint
            </h2>
            <p className="max-w-md text-sm text-emerald-50/90">
              Register once, then enroll and verify your voice with modern biometric security controls.
            </p>
            <div className="inline-flex items-center gap-2 rounded-full bg-white/15 px-3 py-2 text-xs font-medium backdrop-blur">
              <BrandIcon size="xs" />
              Trusted secure onboarding
            </div>
          </div>
        </div>
      </aside>

      <div className="p-5 sm:p-8 lg:p-10">
        <header className="mb-5 space-y-1.5 sm:mb-6 sm:space-y-2">
          <div className="mb-3 flex items-center gap-2.5 lg:hidden">
            <BrandIcon size="sm" />
            <span className="text-xs font-bold uppercase tracking-[0.16em] text-brand-700 dark:text-brand-300">
              Voice Authentication
            </span>
          </div>
          <h1 className="text-2xl font-extrabold tracking-tight text-slate-900 dark:text-slate-100 sm:text-3xl">
            {title}
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 sm:text-base">{subtitle}</p>
        </header>

        {children}
      </div>
    </section>
  )
}

export default AuthCard
