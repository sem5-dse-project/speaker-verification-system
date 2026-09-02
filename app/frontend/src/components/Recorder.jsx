import { useEffect, useMemo, useRef, useState } from 'react'
import { Mic, PauseCircle, PlayCircle, Trash2 } from 'lucide-react'
import PrimaryButton from './PrimaryButton.jsx'
import StatusBadge from './StatusBadge.jsx'
import { floatSamplesToWavBlob } from '../utils/audioWav.js'

const formatSeconds = (seconds) => {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

function Recorder({ onRecordingChange, onRecorderError }) {
  const [status, setStatus] = useState('ready')
  const [seconds, setSeconds] = useState(0)
  const [audioUrl, setAudioUrl] = useState(null)

  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const audioRef = useRef(null)
  const audioContextRef = useRef(null)
  const processorRef = useRef(null)
  const sourceRef = useRef(null)
  const pcmChunksRef = useRef([])
  const sampleRateRef = useRef(16000)
  const recordingRef = useRef(false)

  const hasRecording = useMemo(() => Boolean(audioUrl), [audioUrl])

  const cleanupAudioGraph = () => {
    recordingRef.current = false

    try {
      processorRef.current?.disconnect()
    } catch {
      /* ignore */
    }
    try {
      sourceRef.current?.disconnect()
    } catch {
      /* ignore */
    }

    processorRef.current = null
    sourceRef.current = null

    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {})
      audioContextRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
  }

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
      cleanupAudioGraph()
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
      if (!navigator.mediaDevices?.getUserMedia) {
        onRecorderError('This browser does not support audio recording.')
        return
      }

      const AudioCtx = window.AudioContext || window.webkitAudioContext
      if (!AudioCtx) {
        onRecorderError('Web Audio API is not available in this browser.')
        return
      }

      if (audioUrl) {
        URL.revokeObjectURL(audioUrl)
        setAudioUrl(null)
      }

      onRecordingChange(null)
      onRecorderError('')
      cleanupAudioGraph()

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
        },
      })

      const audioContext = new AudioCtx()
      const source = audioContext.createMediaStreamSource(stream)
      // ScriptProcessor is deprecated but widely supported for PCM capture.
      const processor = audioContext.createScriptProcessor(4096, 1, 1)

      pcmChunksRef.current = []
      sampleRateRef.current = audioContext.sampleRate
      recordingRef.current = true

      processor.onaudioprocess = (event) => {
        if (!recordingRef.current) {
          return
        }
        const input = event.inputBuffer.getChannelData(0)
        pcmChunksRef.current.push(new Float32Array(input))
      }

      source.connect(processor)
      // Keep the graph alive; mute output to avoid feedback.
      const gain = audioContext.createGain()
      gain.gain.value = 0
      processor.connect(gain)
      gain.connect(audioContext.destination)

      streamRef.current = stream
      audioContextRef.current = audioContext
      sourceRef.current = source
      processorRef.current = processor

      setSeconds(0)
      setStatus('recording')
      startTimer()
    } catch {
      cleanupAudioGraph()
      setStatus('ready')
      stopTimer()
      onRecorderError('Microphone access failed. Please allow permissions and try again.')
    }
  }

  const handleStopRecording = () => {
    if (status !== 'recording') {
      return
    }

    stopTimer()
    recordingRef.current = false

    try {
      const chunks = pcmChunksRef.current
      const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0)

      if (totalLength < sampleRateRef.current * 0.5) {
        cleanupAudioGraph()
        setStatus('ready')
        onRecorderError('Recording too short. Please speak for at least half a second.')
        return
      }

      const merged = new Float32Array(totalLength)
      let offset = 0
      for (const chunk of chunks) {
        merged.set(chunk, offset)
        offset += chunk.length
      }

      const audioBlob = floatSamplesToWavBlob(merged, sampleRateRef.current)
      const nextAudioUrl = URL.createObjectURL(audioBlob)

      cleanupAudioGraph()
      setAudioUrl(nextAudioUrl)
      setStatus('complete')
      onRecordingChange({ blob: audioBlob, url: nextAudioUrl })
    } catch (error) {
      cleanupAudioGraph()
      setStatus('ready')
      setAudioUrl(null)
      onRecordingChange(null)
      onRecorderError(error.message || 'Failed to encode WAV recording.')
    }
  }

  const handlePlayRecording = async () => {
    if (audioRef.current && hasRecording) {
      await audioRef.current.play()
    }
  }

  const handleDeleteRecording = () => {
    stopTimer()
    cleanupAudioGraph()

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
    <section className="card flex flex-col items-center gap-6 p-6 text-center sm:p-8">
      <div className="relative grid h-28 w-28 place-items-center rounded-full bg-blue-50 ring-2 ring-blue-200 dark:bg-blue-950/40 dark:ring-blue-900/50">
        {status === 'recording' && (
          <span className="absolute h-full w-full animate-ping rounded-full bg-blue-300/40 dark:bg-blue-500/30" />
        )}
        <Mic className="relative h-12 w-12 text-blue-600 dark:text-blue-400" />
      </div>

      <StatusBadge status={status} />
      <p className="text-3xl font-bold tabular-nums text-slate-900 dark:text-slate-100">{formatSeconds(seconds)}</p>

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
