function ProgressRing({ value, max, label, size = 88 }) {
  const safeMax = Math.max(max, 1)
  const progress = Math.min(Math.max(value / safeMax, 0), 1)
  const stroke = 7
  const radius = (size - stroke) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - progress)

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            strokeWidth={stroke}
            className="stroke-brand-100 dark:stroke-brand-900/50"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="stroke-brand-600 transition-[stroke-dashoffset] duration-500 dark:stroke-brand-400"
          />
        </svg>
        <div className="absolute inset-0 grid place-items-center text-center">
          <span className="text-lg font-extrabold tabular-nums text-slate-900 dark:text-slate-100">
            {value}/{max}
          </span>
        </div>
      </div>
      {label && <p className="text-xs font-semibold uppercase tracking-wide text-subtle">{label}</p>}
    </div>
  )
}

export default ProgressRing
