import { CheckCircle2, Circle, Mic } from 'lucide-react'

const STATUS_MAP = {
  ready: {
    label: 'Ready',
    icon: Circle,
    className: 'bg-slate-100 text-slate-700 ring-1 ring-slate-200',
  },
  recording: {
    label: 'Recording...',
    icon: Mic,
    className: 'bg-rose-100 text-rose-700 ring-1 ring-rose-200',
  },
  complete: {
    label: 'Recording Complete',
    icon: CheckCircle2,
    className: 'bg-emerald-100 text-emerald-700 ring-1 ring-emerald-200',
  },
}

function StatusBadge({ status }) {
  const selected = STATUS_MAP[status] ?? STATUS_MAP.ready
  const Icon = selected.icon

  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-colors ${selected.className}`}
    >
      <Icon className="h-4 w-4" />
      <span>{selected.label}</span>
    </div>
  )
}

export default StatusBadge
