function SkeletonBlock({ className = '' }) {
  return <div className={`skeleton-block ${className}`.trim()} aria-hidden="true" />
}

function DashboardSkeleton() {
  return (
    <div className="space-y-5 sm:space-y-6" aria-busy="true" aria-label="Loading dashboard">
      <div className="space-y-2">
        <SkeletonBlock className="h-4 w-32" />
        <SkeletonBlock className="h-8 w-64 max-w-full" />
        <SkeletonBlock className="h-4 w-80 max-w-full" />
      </div>

      <div className="card grid gap-5 sm:grid-cols-[auto_1fr] sm:items-center">
        <SkeletonBlock className="mx-auto h-24 w-24 rounded-full" />
        <div className="space-y-3">
          <SkeletonBlock className="h-6 w-40" />
          <SkeletonBlock className="h-4 w-full" />
          <SkeletonBlock className="h-4 w-5/6" />
          <SkeletonBlock className="h-11 w-full max-w-xs rounded-xl" />
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <SkeletonBlock className="h-24 rounded-2xl" />
        <SkeletonBlock className="h-24 rounded-2xl" />
        <SkeletonBlock className="h-24 rounded-2xl sm:col-span-2" />
      </div>

      <SkeletonBlock className="h-40 rounded-2xl" />
    </div>
  )
}

export { SkeletonBlock, DashboardSkeleton }
