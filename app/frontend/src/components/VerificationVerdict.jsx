import { AlertTriangle, CheckCircle2, ShieldAlert, ShieldX, VolumeX } from 'lucide-react'

const VERDICT_CONFIG = {
  ACCEPT: {
    title: 'Voice Verified',
    subtitle: 'Your voice matched the enrolled template.',
    icon: CheckCircle2,
    className: 'verdict-success',
  },
  REJECT: {
    title: 'Verification Failed',
    subtitle: 'Your voice did not match the enrolled template.',
    icon: ShieldX,
    className: 'verdict-error',
  },
  REPLAY: {
    title: 'Replay Detected',
    subtitle: 'This sample looks like a recording, not live speech.',
    icon: ShieldAlert,
    className: 'verdict-error',
  },
  SYNTHETIC: {
    title: 'Synthetic Speech Detected',
    subtitle: 'The sample failed synthetic speech screening.',
    icon: ShieldAlert,
    className: 'verdict-error',
  },
  UNCERTAIN: {
    title: 'Inconclusive Result',
    subtitle: 'Please re-record in a quiet room and try again.',
    icon: AlertTriangle,
    className: 'verdict-warning',
  },
  NO_SPEECH: {
    title: 'No Speech Detected',
    subtitle: 'Speak clearly toward the microphone and retry.',
    icon: VolumeX,
    className: 'verdict-warning',
  },
}

function VerificationVerdict({ decision, score }) {
  if (!decision) {
    return null
  }

  const config = VERDICT_CONFIG[decision] || VERDICT_CONFIG.UNCERTAIN
  const Icon = config.icon
  const scoreText = typeof score === 'number' ? score.toFixed(3) : null

  return (
    <div className={`verdict-banner ${config.className}`}>
      <div className="verdict-banner-icon">
        <Icon className="h-7 w-7" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-lg font-extrabold tracking-tight">{config.title}</p>
        <p className="mt-0.5 text-sm opacity-90">{config.subtitle}</p>
        {scoreText && (
          <p className="mt-2 text-xs font-semibold uppercase tracking-wide opacity-80">
            Similarity score: {scoreText}
          </p>
        )}
      </div>
    </div>
  )
}

export default VerificationVerdict
