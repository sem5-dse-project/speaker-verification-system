import { useMemo, useState } from 'react'
import { CheckCircle2, AlertCircle, Circle, ArrowLeft } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import PageShell from '../components/PageShell.jsx'
import SentenceCard from '../components/SentenceCard.jsx'
import Recorder from '../components/Recorder.jsx'
import PrimaryButton from '../components/PrimaryButton.jsx'
import api from '../services/api.js'

const REQUIRED_SAMPLES = 3

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

const pickRandomSentence = (exclude = null) => {
  let next = SENTENCES[Math.floor(Math.random() * SENTENCES.length)]
  if (SENTENCES.length > 1 && exclude) {
    while (next === exclude) {
      next = SENTENCES[Math.floor(Math.random() * SENTENCES.length)]
    }
  }
  return next
}

const slotClassName = (done, active) => {
  if (done) {
    return 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:ring-emerald-900/50'
  }
  if (active) {
    return 'bg-brand-50 text-brand-800 ring-1 ring-brand-200 dark:bg-brand-950/40 dark:text-brand-300 dark:ring-brand-900/50'
  }
  return 'bg-slate-50 text-slate-400 ring-1 ring-slate-200 dark:bg-slate-800/60 dark:text-slate-500 dark:ring-slate-700'
}

function Enrollment() {
  const navigate = useNavigate()
  const [sentence, setSentence] = useState(() => pickRandomSentence())
  const [samples, setSamples] = useState([])
  const [currentRecording, setCurrentRecording] = useState(null)
  const [recorderKey, setRecorderKey] = useState(0)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [statusMessage, setStatusMessage] = useState({ type: '', text: '' })

  const sampleCount = samples.length
  const allSamplesReady = sampleCount >= REQUIRED_SAMPLES
  const canSaveCurrent = Boolean(currentRecording) && !allSamplesReady

  const progressLabel = useMemo(
    () => `Sample ${Math.min(sampleCount + 1, REQUIRED_SAMPLES)} of ${REQUIRED_SAMPLES}`,
    [sampleCount],
  )

  const handleGenerateSentence = () => {
    setSentence((previous) => pickRandomSentence(previous))
  }

  const resetRecorder = () => {
    setCurrentRecording(null)
    setRecorderKey((key) => key + 1)
    setSentence((previous) => pickRandomSentence(previous))
  }

  const handleSaveSample = () => {
    if (!canSaveCurrent) {
      return
    }

    setSamples((previous) => [
      ...previous,
      {
        blob: currentRecording.blob,
        url: currentRecording.url,
        sentence,
      },
    ])
    setStatusMessage({
      type: '',
      text: '',
    })
    resetRecorder()
  }

  const handleRemoveSample = (index) => {
    setSamples((previous) => {
      const next = [...previous]
      const [removed] = next.splice(index, 1)
      if (removed?.url) {
        URL.revokeObjectURL(removed.url)
      }
      return next
    })
    setStatusMessage({ type: '', text: '' })
  }

  const handleEnrollVoice = async () => {
    if (!allSamplesReady) {
      setStatusMessage({
        type: 'error',
        text: `Please record all ${REQUIRED_SAMPLES} enrollment samples first.`,
      })
      return
    }

    setIsSubmitting(true)
    setStatusMessage({ type: '', text: '' })

    try {
      await api.post('/voice/enroll/reset')

      for (let index = 0; index < samples.length; index += 1) {
        const formData = new FormData()
        const file = new File(
          [samples[index].blob],
          `enroll_${index + 1}_${Date.now()}.wav`,
          { type: 'audio/wav' },
        )
        formData.append('audio', file)
        await api.post('/voice/enroll', formData)
      }

      setStatusMessage({
        type: 'success',
        text: `Enrollment complete. Template saved from ${REQUIRED_SAMPLES} WAV samples.`,
      })
      setSamples([])
      resetRecorder()
    } catch (error) {
      setStatusMessage({
        type: 'error',
        text:
          error.response?.data?.error ||
          error.response?.data?.message ||
          'Failed to upload enrollment recordings.',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <PageShell>
      <div className="mx-auto max-w-4xl space-y-6">
        <div>
          <button type="button" onClick={() => navigate('/dashboard')} className="btn-secondary">
            <ArrowLeft className="h-4 w-4" />
            Back
          </button>
        </div>

        <header className="space-y-2">
          <h1 className="heading-1">Voice Enrollment</h1>
          <p className="text-base text-muted sm:text-lg">
            Record {REQUIRED_SAMPLES} different voice samples to build a stronger enrollment template.
          </p>
        </header>

        <section className="card p-6 sm:p-8">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-subtle">
            Instructions
          </h2>
          <p className="text-body">
            Record {REQUIRED_SAMPLES} samples. Use a different sentence for each if possible.
            Speak clearly in a quiet room. After each recording, click <strong>Save sample</strong>,
            then finish with <strong>Complete enrollment</strong>.
          </p>
        </section>

        <section className="card p-5">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-sm font-semibold text-body">{progressLabel}</p>
            <p className="text-sm text-subtle">
              Saved: {sampleCount}/{REQUIRED_SAMPLES}
            </p>
          </div>
          <div className="flex gap-2">
            {Array.from({ length: REQUIRED_SAMPLES }).map((_, index) => {
              const done = index < sampleCount
              return (
                <div
                  key={`slot-${index}`}
                  className={`flex flex-1 items-center justify-center gap-2 rounded-xl px-3 py-3 text-sm font-medium ${slotClassName(done, index === sampleCount)}`}
                >
                  {done ? <CheckCircle2 className="h-4 w-4" /> : <Circle className="h-4 w-4" />}
                  Sample {index + 1}
                </div>
              )
            })}
          </div>

          {samples.length > 0 && (
            <ul className="mt-4 space-y-2">
              {samples.map((sample, index) => (
                <li
                  key={`saved-${index}`}
                  className="list-item flex flex-wrap items-center justify-between gap-2 bg-slate-50 dark:bg-slate-800/60"
                >
                  <span className="line-clamp-1">
                    #{index + 1}: {sample.sentence}
                  </span>
                  <button
                    type="button"
                    onClick={() => handleRemoveSample(index)}
                    className="text-rose-600 hover:underline dark:text-rose-400"
                    disabled={isSubmitting}
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        {!allSamplesReady && (
          <>
            <SentenceCard sentence={sentence} onGenerateSentence={handleGenerateSentence} />

            <Recorder
              key={recorderKey}
              onRecordingChange={setCurrentRecording}
              onRecorderError={(message) =>
                setStatusMessage(
                  message ? { type: 'error', text: message } : { type: '', text: '' },
                )
              }
            />

            <PrimaryButton
              type="button"
              onClick={handleSaveSample}
              disabled={!canSaveCurrent || isSubmitting}
              className="w-full bg-brand-800 py-3 hover:bg-brand-700 dark:bg-brand-900 dark:hover:bg-brand-800"
            >
              Save sample {sampleCount + 1}
            </PrimaryButton>
          </>
        )}

        <div className="space-y-3">
          <PrimaryButton
            type="button"
            onClick={handleEnrollVoice}
            disabled={!allSamplesReady || isSubmitting}
            className="w-full py-4 text-lg"
          >
            {isSubmitting
              ? 'Uploading samples...'
              : `Complete enrollment (${REQUIRED_SAMPLES} samples)`}
          </PrimaryButton>

          <div className="status-panel">
            {statusMessage.type === 'success' && (
              <p className="flex items-center gap-2 text-emerald-700 dark:text-emerald-300">
                <CheckCircle2 className="h-4 w-4" />
                {statusMessage.text}
              </p>
            )}

            {statusMessage.type === 'error' && (
              <p className="flex items-center gap-2 text-rose-700 dark:text-rose-300">
                <AlertCircle className="h-4 w-4" />
                {statusMessage.text}
              </p>
            )}

            {!statusMessage.type && (
              <p className="text-subtle">
                Record and save {REQUIRED_SAMPLES} samples, then complete enrollment.
              </p>
            )}
          </div>
        </div>
      </div>
    </PageShell>
  )
}

export default Enrollment
