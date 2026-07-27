import { useEffect, useMemo, useRef, useState } from 'react'
import { Mic, PauseCircle, PlayCircle, Trash2 } from 'lucide-react'
import PrimaryButton from './PrimaryButton.jsx'
import StatusBadge from './StatusBadge.jsx'

const formatSeconds = (seconds) => {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

function Recorder({ onRecordingChange, onRecorderError }) {
  const [status, setStatus] = useState('ready')
  const [seconds, setSeconds] = useState(0)
  const [audioUrl, setAudioUrl] = useState(null)

  const mediaRecorderRef = useRef(null)
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const chunksRef = useRef([])
  const audioRef = useRef(null)

  const hasRecording = useMemo(() => Boolean(audioUrl), [audioUrl])

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
      }

      if (audioUrl) {
        URL.revokeObjectURL(audioUrl)
      }
    }
  }, [audioUrl])

  const startTimer = () => {
    timerRef.current = setInterval(() => {
      setSeconds((previous) => previous + 1)
    }, 1000)
  }

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  const handleStartRecording = async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        onRecorderError('This browser does not support audio recording.')
        return
      }

      if (audioUrl) {
        URL.revokeObjectURL(audioUrl)
        setAudioUrl(null)
      }

      onRecordingChange(null)
      onRecorderError('')

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)

      streamRef.current = stream
      mediaRecorderRef.current = recorder
      chunksRef.current = []
      setSeconds(0)
      setStatus('recording')
      startTimer()

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      recorder.onstop = () => {
        stopTimer()
        setStatus('complete')

        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const nextAudioUrl = URL.createObjectURL(audioBlob)

        setAudioUrl(nextAudioUrl)
        onRecordingChange({ blob: audioBlob, url: nextAudioUrl })

        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop())
          streamRef.current = null
        }

        mediaRecorderRef.current = null
      }

      recorder.start()
    } catch {
      setStatus('ready')
      stopTimer()
      onRecorderError('Microphone access failed. Please allow permissions and try again.')
    }
  }

  const handleStopRecording = () => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
  }

  const handlePlayRecording = async () => {
    if (audioRef.current && hasRecording) {
      await audioRef.current.play()
    }
  }

  const handleDeleteRecording = () => {
    stopTimer()

    if (audioUrl) {
      URL.revokeObjectURL(audioUrl)
    }

    setAudioUrl(null)
    setSeconds(0)
    setStatus('ready')
    onRecordingChange(null)
    onRecorderError('')
  }

  return (
    <section className="flex flex-col items-center gap-6 rounded-2xl bg-white p-6 text-center shadow-sm ring-1 ring-slate-200 sm:p-8">
      <div className="relative grid h-28 w-28 place-items-center rounded-full bg-blue-50 ring-2 ring-blue-200">
        {status === 'recording' && (
          <span className="absolute h-full w-full animate-ping rounded-full bg-blue-300/40" />
        )}
        <Mic className="relative h-12 w-12 text-blue-600" />
      </div>

      <StatusBadge status={status} />
      <p className="text-3xl font-bold tabular-nums text-slate-900">{formatSeconds(seconds)}</p>

      <div className="flex flex-wrap justify-center gap-3">
        <PrimaryButton
          type="button"
          onClick={handleStartRecording}
          disabled={status === 'recording'}
        >
          <Mic className="mr-2 h-4 w-4" />
          Start Recording
        </PrimaryButton>

        <PrimaryButton
          type="button"
          onClick={handleStopRecording}
          disabled={status !== 'recording'}
          className="bg-slate-700 shadow-slate-200 hover:bg-slate-600"
        >
          <PauseCircle className="mr-2 h-4 w-4" />
          Stop Recording
        </PrimaryButton>

        <PrimaryButton
          type="button"
          onClick={handlePlayRecording}
          disabled={!hasRecording || status === 'recording'}
          className="bg-slate-700 shadow-slate-200 hover:bg-slate-600"
        >
          <PlayCircle className="mr-2 h-4 w-4" />
          Play Recording
        </PrimaryButton>

        <PrimaryButton
          type="button"
          onClick={handleDeleteRecording}
          disabled={!hasRecording || status === 'recording'}
          className="bg-slate-700 shadow-slate-200 hover:bg-slate-600"
        >
          <Trash2 className="mr-2 h-4 w-4" />
          Delete Recording
        </PrimaryButton>
      </div>

      <audio ref={audioRef} src={audioUrl ?? undefined} className="hidden" />
    </section>
  )
}

export default Recorder
