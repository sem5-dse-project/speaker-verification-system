import { useMemo, useState } from 'react'
import { CheckCircle2, AlertCircle } from 'lucide-react'
import SentenceCard from '../components/SentenceCard.jsx'
import Recorder from '../components/Recorder.jsx'
import PrimaryButton from '../components/PrimaryButton.jsx'
import api from '../services/api.js'

const SENTENCES = [
  'The quick brown fox jumps over the lazy dog.',
  'Please verify my identity with this spoken sentence.',
  'Blue skies often follow a quiet rainy morning.',
  'Consistent practice improves confidence and clarity.',
  'Security begins with careful attention to detail.',
  'Today I will speak clearly for voice enrollment.',
  'Modern systems rely on accurate user authentication.',
  'A calm voice in a quiet room improves quality.',
  'Reliable verification depends on clean audio input.',
  'My voice is unique and ready for enrollment.',
  'Clear pronunciation helps the model learn better.',
]

const pickRandomSentence = () => {
  const index = Math.floor(Math.random() * SENTENCES.length)
  return SENTENCES[index]
}

function Enrollment() {
  const [sentence, setSentence] = useState(() => pickRandomSentence())
  const [recording, setRecording] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [statusMessage, setStatusMessage] = useState({ type: '', text: '' })

  const hasRecording = useMemo(() => Boolean(recording), [recording])

  const handleGenerateSentence = () => {
    setSentence((previous) => {
      let nextSentence = previous
      while (nextSentence === previous && SENTENCES.length > 1) {
        nextSentence = pickRandomSentence()
      }
      return nextSentence
    })
  }

  const handleEnrollVoice = async () => {
    if (!hasRecording) {
      return
    }

    setIsSubmitting(true)
    setStatusMessage({ type: '', text: '' })

    try {
      const formData = new FormData()
      const wavLikeFile = new File([recording.blob], `enroll_${Date.now()}.wav`, {
        type: 'audio/wav',
      })
      formData.append('audio', wavLikeFile)

      const response = await api.post('/voice/enroll', formData)
      setStatusMessage({
        type: 'success',
        text: response.data?.message || 'Voice enrollment complete.',
      })
    } catch (error) {
      setStatusMessage({
        type: 'error',
        text: error.response?.data?.message || 'Failed to upload enrollment recording.',
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
            Voice Enrollment
          </h1>
          <p className="text-base text-slate-600 sm:text-lg">
            Enroll your voice by reading the prompted sentence.
          </p>
        </header>

        <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200 sm:p-8">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
            Instructions
          </h2>
          <p className="text-slate-700">
            To improve enrollment quality, please read the sentence shown below exactly as it appears. Speak clearly in a quiet environment and avoid background noise.
          </p>
        </section>

        <SentenceCard
          sentence={sentence}
          onGenerateSentence={handleGenerateSentence}
        />

        <Recorder
          onRecordingChange={setRecording}
          onRecorderError={(message) =>
            setStatusMessage(message ? { type: 'error', text: message } : { type: '', text: '' })
          }
        />

        <div className="space-y-3">
          <PrimaryButton
            type="button"
            onClick={handleEnrollVoice}
            disabled={!hasRecording || isSubmitting}
            className="w-full py-4 text-lg"
          >
            {isSubmitting ? 'Enrolling...' : 'Enroll Voice'}
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

            {!statusMessage.type && (
              <p className="text-slate-500">Status messages will appear here.</p>
            )}
          </div>
        </div>
      </div>
    </main>
  )
}

export default Enrollment
