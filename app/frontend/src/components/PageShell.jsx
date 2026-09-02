import AmbientBackground from './AmbientBackground.jsx'
import AppNav from './AppNav.jsx'
import AppFooter from './AppFooter.jsx'
import ThemeToggle from './ThemeToggle.jsx'

function PageShell({
  variant = 'app',
  children,
  className = '',
  narrow = false,
  showNav = false,
  showFooter = true,
}) {
  const variantClass =
    variant === 'auth' ? 'page-auth' : variant === 'admin' ? 'page-admin' : 'page-app'
  const contentClass = [
    narrow ? 'page-content-narrow' : 'page-content',
    showNav ? 'page-content-with-nav' : '',
  ]
    .filter(Boolean)
    .join(' ')

  const footerVariant = variant === 'auth' ? 'auth' : variant === 'admin' ? 'admin' : 'app'
  const showFloatingTheme = !showNav

  return (
    <main className={`${variantClass} ${className}`.trim()}>
      <AmbientBackground />

      {showNav && <AppNav />}

      {showFloatingTheme && (
        <div className="fixed right-[max(0.75rem,env(safe-area-inset-right))] top-[max(0.75rem,env(safe-area-inset-top))] z-30 sm:right-6 sm:top-6">
          <ThemeToggle />
        </div>
      )}

      <div className={`${contentClass} flex-1`}>{children}</div>

      {showFooter && <AppFooter variant={footerVariant} withBottomNav={showNav} />}
    </main>
  )
}

export default PageShell
