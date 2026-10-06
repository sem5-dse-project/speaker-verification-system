import { useState } from 'react'
import { ShieldCheck } from 'lucide-react'
import PageShell from '../components/PageShell.jsx'
import PageHeader from '../components/PageHeader.jsx'
import SentenceCard from '../components/SentenceCard.jsx'
import Recorder from '../components/Recorder.jsx'
import PrimaryButton from '../components/PrimaryButton.jsx'
import VerificationVerdict from '../components/VerificationVerdict.jsx'
import VerificationStepper from '../components/VerificationStepper.jsx'
import api from '../services/api.js'
import { formatVerificationResult } from '../utils/verificationResult.js'

const SENTENCES = [
  'The quick brown fox jumps over the lazy dog.',
  'Please verify my identity with this spoken sentence.',
  'Blue skies often follow a quiet rainy morning.',
  'Consistent practice improves confidence and clarity.',
  'Security begins with careful attention to detail.',
  'Today I will speak clearly for voice verification.',
  'Modern systems rely on accurate user authentication.',
  'A calm voice in a quiet room improves quality.',
  'Reliable verification depends on clean audio input.',
  'My voice is unique and ready for verification.',
  'Clear pronunciation helps the check hear me clearly.',
]

const pickRandomSentence = (exclude = null) => {
  let next = SENTENCES[Math.floor(Math.random() * SENTENCES.length)]
  if (SENTENCES.length > 1 && exclude) {
    while (next === exclude) {
      next = SENTENCES[Math.floor(Math.random() * SENTENCES.length)]
    }
  }
  return next
}

function Verification() {
  const [sentence, setSentence] = useState(() => pickRandomSentence())
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
          subtitle="Read the sentence aloud, then submit the recording for replay screening and speaker matching."
        />

        <SentenceCard
          sentence={sentence}
          onGenerateSentence={() => setSentence((previous) => pickRandomSentence(previous))}
        />

        <Recorder
          readsSentence
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
