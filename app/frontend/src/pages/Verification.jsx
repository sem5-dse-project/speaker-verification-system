import { useState } from 'react'
import { AlertCircle, CheckCircle2, ArrowLeft } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import Recorder from '../components/Recorder.jsx'
import PrimaryButton from '../components/PrimaryButton.jsx'
import api from '../services/api.js'
import { formatVerificationResult } from '../utils/verificationResult.js'

const STATUS_STYLES = {
  success: 'border-emerald-200 bg-emerald-50 text-emerald-800',
  warning: 'border-amber-200 bg-amber-50 text-amber-800',
  error: 'border-rose-200 bg-rose-50 text-rose-800',
  neutral: 'border-slate-200 bg-slate-50 text-slate-700',
}

const metricClassName =
  'rounded-lg border border-slate-200 bg-white px-3 py-2 text-slate-700 shadow-sm'

function Verification() {
  const navigate = useNavigate()
  const [recording, setRecording] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [statusMessage, setStatusMessage] = useState({ type: '', text: '', details: [] })

  const handleVerifyVoice = async () => {
    if (!recording?.blob) {
      return
    }

    setIsSubmitting(true)
    setStatusMessage({ type: '', text: '' })

    try {
      const formData = new FormData()
      const wavLikeFile = new File([recording.blob], `verify_1_${Date.now()}.wav`, {
        type: 'audio/wav',
      })
      formData.append('audio', wavLikeFile)

      const response = await api.post('/voice/verify', formData)
      const formatted = formatVerificationResult(
        response.data?.result,
        response.data?.message || 'Verification complete.',
      )
      setStatusMessage(formatted)
    } catch (error) {
      setStatusMessage({
        type: 'error',
        text: error.response?.data?.message || 'Failed to upload verification sample.',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-100 to-white px-4 py-10 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-4xl space-y-6">
        <div>
          <button
            type="button"
            onClick={() => navigate('/dashboard')}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </button>
        </div>

        <header className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
            Voice Verification
          </h1>
          <p className="text-base text-slate-600 sm:text-lg">
            Record your voice and submit it for verification. Quiet clips are rejected, clear replay
            is blocked, uncertain clips ask you to re-record, then live speech is matched to your
            enrolled template.
          </p>
        </header>

        <Recorder
          onRecordingChange={setRecording}
          onRecorderError={(message) =>
            setStatusMessage(message ? { type: 'error', text: message } : { type: '', text: '' })
          }
        />

        <div className="space-y-3">
          <PrimaryButton
            type="button"
            onClick={handleVerifyVoice}
            disabled={!recording?.blob || isSubmitting}
            className="w-full py-4 text-lg"
          >
            {isSubmitting ? 'Submitting...' : 'Verify Voice'}
          </PrimaryButton>

          <div className="min-h-12 rounded-xl border border-dashed border-slate-300 bg-slate-50 px-4 py-3 text-sm">
            {statusMessage.type === 'success' && (
              <p className="flex items-center gap-2 text-emerald-700">
                <CheckCircle2 className="h-4 w-4" />
                {statusMessage.text}
              </p>
            )}

            {statusMessage.type === 'error' && (
              <p className="flex items-center gap-2 text-rose-700">
                <AlertCircle className="h-4 w-4" />
                {statusMessage.text}
              </p>
            )}

            {statusMessage.type === 'warning' && (
              <p className="flex items-center gap-2 text-amber-700">
                <AlertCircle className="h-4 w-4" />
                {statusMessage.text}
              </p>
            )}

            {!statusMessage.type && <p className="text-slate-500">Status messages will appear here.</p>}
          </div>

          {statusMessage.details?.length > 0 && (
            <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="mb-4 flex flex-wrap items-start justify-between gap-2">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">Verification metrics</h2>
                  <p className="text-sm text-slate-600">
                    Stage-by-stage breakdown for the detected voice sample.
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                {statusMessage.details.map((stage, index) => (
                  <article
                    key={`${stage.label}-${index}`}
                    className="rounded-2xl border border-slate-200 bg-slate-50 p-4"
                  >
                    <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <p className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                          Stage {index + 1}
                        </p>
                        <h3 className="text-base font-semibold text-slate-900">{stage.label}</h3>
                        <p className="mt-1 text-sm text-slate-600">{stage.summary}</p>
                      </div>

                      <span
                        className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${STATUS_STYLES[stage.status] || STATUS_STYLES.neutral}`}
                      >
                        {stage.status}
                      </span>
                    </div>

                    {stage.metrics?.length > 0 && (
                      <dl className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                        {stage.metrics.map((metric) => (
                          <div key={`${stage.label}-${metric.label}`} className={metricClassName}>
                            <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
                              {metric.label}
                            </dt>
                            <dd className="mt-1 text-sm font-semibold text-slate-900">
                              {metric.value}
                            </dd>
                          </div>
                        ))}
                      </dl>
                    )}
                  </article>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>
    </main>
  )
}

export default Verification
