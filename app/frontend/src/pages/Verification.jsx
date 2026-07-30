import { useState } from 'react'
import { AlertCircle, CheckCircle2 } from 'lucide-react'
import Recorder from '../components/Recorder.jsx'
import PrimaryButton from '../components/PrimaryButton.jsx'
import api from '../services/api.js'

function Verification() {
  const [recording, setRecording] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [statusMessage, setStatusMessage] = useState({ type: '', text: '' })

  const handleVerifyVoice = async () => {
    if (!recording?.blob) {
      return
    }

    setIsSubmitting(true)
    setStatusMessage({ type: '', text: '' })

    try {
      const formData = new FormData()
      const wavLikeFile = new File([recording.blob], `verify_${Date.now()}.wav`, {
        type: 'audio/wav',
      })
      formData.append('audio', wavLikeFile)

      const response = await api.post('/voice/verify', formData)

      const result = response.data?.result
      const decision =
        result?.decision ||
        (result?.accepted === true
          ? 'ACCEPT'
          : result?.accepted === false
            ? 'REJECT'
            : null)
      const scoreText =
        typeof result?.score === 'number' ? ` (score ${result.score.toFixed(3)})` : ''

      setStatusMessage({
        type: decision === 'REJECT' ? 'error' : 'success',
        text: decision
          ? `Verification ${decision}${scoreText}`
          : response.data?.message || 'Verification complete.',
      })
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
        <header className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
            Voice Verification
          </h1>
          <p className="text-base text-slate-600 sm:text-lg">
            Record your voice and submit it for verification.
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

            {!statusMessage.type && <p className="text-slate-500">Status messages will appear here.</p>}
          </div>
        </div>
      </div>
    </main>
  )
}

export default Verification
