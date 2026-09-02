import ThemeToggle from './ThemeToggle.jsx'

function PageShell({ variant = 'app', children, className = '', narrow = false }) {
  const variantClass =
    variant === 'auth' ? 'page-auth' : variant === 'admin' ? 'page-admin' : 'page-app'
  const contentClass = narrow ? 'page-content-narrow' : 'page-content'

  return (
    <main className={`${variantClass} ${className}`.trim()}>
      <div className="fixed right-[max(0.75rem,env(safe-area-inset-right))] top-[max(0.75rem,env(safe-area-inset-top))] z-30 sm:right-6 sm:top-6">
        <ThemeToggle />
      </div>
      <div className={contentClass}>{children}</div>
    </main>
  )
}

export default PageShell
