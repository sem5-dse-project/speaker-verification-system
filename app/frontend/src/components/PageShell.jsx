import AmbientBackground from './AmbientBackground.jsx'
import AppNav from './AppNav.jsx'
import ThemeToggle from './ThemeToggle.jsx'

function PageShell({ variant = 'app', children, className = '', narrow = false, showNav = false }) {
  const variantClass =
    variant === 'auth' ? 'page-auth' : variant === 'admin' ? 'page-admin' : 'page-app'
  const contentClass = [
    narrow ? 'page-content-narrow' : 'page-content',
    showNav ? 'page-content-with-nav' : '',
  ]
    .filter(Boolean)
    .join(' ')

  const showAmbient = variant !== 'auth'
  const showFloatingTheme = !showNav

  return (
    <main className={`${variantClass} ${className}`.trim()}>
      {showAmbient && <AmbientBackground />}

      {showNav && <AppNav />}

      {showFloatingTheme && (
        <div className="fixed right-[max(0.75rem,env(safe-area-inset-right))] top-[max(0.75rem,env(safe-area-inset-top))] z-30 sm:right-6 sm:top-6">
          <ThemeToggle />
        </div>
      )}

      <div className={contentClass}>{children}</div>
    </main>
  )
}

export default PageShell
