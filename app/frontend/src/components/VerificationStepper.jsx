import { CheckCircle2, Circle, XCircle } from 'lucide-react'

const STATUS_STYLES = {
  success: 'stepper-dot-success',
  warning: 'stepper-dot-warning',
  error: 'stepper-dot-error',
  neutral: 'stepper-dot-neutral',
}

const STATUS_ICONS = {
  success: CheckCircle2,
  warning: Circle,
  error: XCircle,
  neutral: Circle,
}

function VerificationStepper({ stages }) {
  if (!stages?.length) {
    return null
  }

  return (
    <ol className="verification-stepper">
      {stages.map((stage, index) => {
        const status = stage.status || 'neutral'
        const Icon = STATUS_ICONS[status] || Circle
        const isLast = index === stages.length - 1

        return (
          <li key={`${stage.label}-${index}`} className="verification-stepper-item">
            <div className="verification-stepper-track">
              <span className={`verification-stepper-dot ${STATUS_STYLES[status] || STATUS_STYLES.neutral}`}>
                <Icon className="h-4 w-4" />
              </span>
              {!isLast && <span className="verification-stepper-line" aria-hidden="true" />}
            </div>

            <article className="verification-stepper-body card-muted">
              <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-subtle">
                    Stage {index + 1}
                  </p>
                  <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">
                    {stage.label}
                  </h3>
                </div>
                <span className={`stepper-badge stepper-badge-${status}`}>{stage.status}</span>
              </div>
              <p className="text-sm text-muted">{stage.summary}</p>

              {stage.metrics?.length > 0 && (
                <dl className="mt-3 grid gap-2 sm:grid-cols-2">
                  {stage.metrics.map((metric) => (
                    <div key={`${stage.label}-${metric.label}`} className="metric-chip">
                      <dt className="text-xs font-medium uppercase tracking-wide text-subtle">
                        {metric.label}
                      </dt>
                      <dd className="mt-1 text-sm font-semibold text-slate-900 dark:text-slate-100">
                        {metric.value}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}
            </article>
          </li>
        )
      })}
    </ol>
  )
}

export default VerificationStepper
