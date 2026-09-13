function PageHeader({ icon: Icon, title, subtitle, action }) {
  return (
    <header className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
      <div className="flex min-w-0 items-start gap-3">
        {Icon && (
          <span className="page-header-icon">
            <Icon className="h-5 w-5" />
          </span>
        )}
        <div className="min-w-0 space-y-1">
          <h1 className="heading-1">{title}</h1>
          {subtitle && <p className="text-sm text-muted sm:text-base">{subtitle}</p>}
        </div>
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </header>
  )
}

export default PageHeader
