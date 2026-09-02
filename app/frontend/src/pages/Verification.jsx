import { useState } from 'react'
import { ShieldCheck } from 'lucide-react'
import PageShell from '../components/PageShell.jsx'
import PageHeader from '../components/PageHeader.jsx'
import Recorder from '../components/Recorder.jsx'
import PrimaryButton from '../components/PrimaryButton.jsx'
import VerificationVerdict from '../components/VerificationVerdict.jsx'
import VerificationStepper from '../components/VerificationStepper.jsx'
import api from '../services/api.js'
import { formatVerificationResult } from '../utils/verificationResult.js'

function Verification() {
  const [recording, setRecording] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [statusMessage, setStatusMessage] = useState({
    type: '',
    text: '',
    details: [],
    decision: null,
    score: null,
  })

  const handleVerifyVoice = async () => {
    if (!recording?.blob) {
      return
    }

    setIsSubmitting(true)
    setStatusMessage({ type: '', text: '', details: [], decision: null, score: null })

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
        details: [],
        decision: null,
        score: null,
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <PageShell narrow showNav>
      <div className="space-y-5 sm:space-y-6">
        <PageHeader
          icon={ShieldCheck}
          title="Voice Verification"
          subtitle="Record your voice for a multi-stage security scan: speech detection, replay screening, synthetic checks, then speaker matching."
        />

        <Recorder
          onRecordingChange={setRecording}
          onRecorderError={(message) =>
            setStatusMessage(
              message
                ? { type: 'error', text: message, details: [], decision: null, score: null }
                : { type: '', text: '', details: [], decision: null, score: null },
            )
          }
        />

        <div className="space-y-3">
          <PrimaryButton
            type="button"
            onClick={handleVerifyVoice}
            disabled={!recording?.blob || isSubmitting}
            className="w-full py-4 text-lg sm:w-full"
          >
            {isSubmitting ? 'Scanning voice sample...' : 'Verify Voice'}
          </PrimaryButton>

          {statusMessage.decision && (
            <VerificationVerdict decision={statusMessage.decision} score={statusMessage.score} />
          )}

          {!statusMessage.decision && (
            <div className="status-panel">
              <p className="text-subtle">
                Results appear here after verification — including replay and speaker-match stages.
              </p>
            </div>
          )}

          {statusMessage.details?.length > 0 && (
            <section className="card">
              <div className="mb-4">
                <h2 className="heading-2">Security pipeline</h2>
                <p className="text-sm text-muted">
                  Stage-by-stage breakdown of how your sample was evaluated.
                </p>
              </div>
              <VerificationStepper stages={statusMessage.details} />
            </section>
          )}
        </div>
      </div>
    </PageShell>
  )
}

export default Verification
