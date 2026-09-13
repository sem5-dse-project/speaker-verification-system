import { ChevronRight } from 'lucide-react'
import { Link } from 'react-router-dom'

function FeatureTile({ to, icon: Icon, title, description, variant = 'primary' }) {
  return (
    <Link to={to} className={`feature-tile group feature-tile-${variant}`}>
      <span className="feature-tile-icon">
        <Icon className="h-5 w-5" />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-base font-bold text-slate-900 dark:text-slate-100">{title}</span>
        <span className="mt-0.5 block text-sm text-muted">{description}</span>
      </span>
      <ChevronRight className="h-5 w-5 shrink-0 text-brand-600 opacity-70 transition group-hover:translate-x-0.5 group-hover:opacity-100 dark:text-brand-400" />
    </Link>
  )
}

export default FeatureTile
