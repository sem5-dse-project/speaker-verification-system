import ThemeToggle from './ThemeToggle.jsx'

function PageShell({ variant = 'app', children, className = '' }) {
  const variantClass =
    variant === 'auth' ? 'page-auth' : variant === 'admin' ? 'page-admin' : 'page-app'

  return (
    <main className={`${variantClass} ${className}`.trim()}>
      <div className="pointer-events-none absolute right-4 top-4 z-30 sm:right-6 sm:top-6">
        <div className="pointer-events-auto">
          <ThemeToggle />
        </div>
      </div>
      {children}
    </main>
  )
}

export default PageShell
