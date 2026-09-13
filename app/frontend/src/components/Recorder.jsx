import { useEffect, useMemo, useRef, useState } from 'react'
import { Mic, PauseCircle, PlayCircle, Trash2 } from 'lucide-react'
import PrimaryButton from './PrimaryButton.jsx'
import StatusBadge from './StatusBadge.jsx'
import AudioWaveform, { BAR_COUNT } from './AudioWaveform.jsx'
import { floatSamplesToWavBlob } from '../utils/audioWav.js'

const formatSeconds = (seconds) => {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

const emptyLevels = () => Array(BAR_COUNT).fill(0.08)

function Recorder({ onRecordingChange, onRecorderError }) {
  const [status, setStatus] = useState('ready')
  const [seconds, setSeconds] = useState(0)
  const [audioUrl, setAudioUrl] = useState(null)
  const [levels, setLevels] = useState(emptyLevels)

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
  const isRecording = status === 'recording'

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

  const pushLevel = (rms) => {
    const normalized = Math.min(1, Math.max(0.08, rms * 10))
    setLevels((previous) => [...previous.slice(1), normalized])
  }

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
      setLevels(emptyLevels())

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
        },
      })

      const audioContext = new AudioCtx()
      const source = audioContext.createMediaStreamSource(stream)
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

        let sum = 0
        for (let index = 0; index < input.length; index += 1) {
          sum += input[index] * input[index]
        }
        pushLevel(Math.sqrt(sum / input.length))
      }

      source.connect(processor)
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
      setLevels(emptyLevels())
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
        setLevels(emptyLevels())
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
      setLevels(emptyLevels())
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
    setLevels(emptyLevels())
    onRecordingChange(null)
    onRecorderError('')
  }

  return (
    <section className="card flex flex-col items-center gap-5 p-4 text-center sm:gap-6 sm:p-8">
      <div className="relative w-full max-w-md">
        <AudioWaveform levels={levels} active={isRecording || hasRecording} />
        <div className="absolute left-1/2 top-1/2 grid h-20 w-20 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-brand-50 ring-2 ring-brand-200 sm:h-24 sm:w-24 dark:bg-surface-800 dark:ring-brand-900/35">
          {isRecording && (
            <span className="absolute h-full w-full animate-ping rounded-full bg-brand-300/40 dark:bg-brand-500/30" />
          )}
          <Mic className="relative h-9 w-9 text-brand-600 sm:h-10 sm:w-10 dark:text-brand-400" />
        </div>
      </div>

      <StatusBadge status={status} />
      <p className="text-2xl font-bold tabular-nums text-slate-900 sm:text-3xl dark:text-slate-100">
        {formatSeconds(seconds)}
      </p>

      <p className="max-w-sm text-xs text-subtle sm:text-sm">
        {isRecording
          ? 'Speak naturally — the waveform reacts to your voice level.'
          : hasRecording
            ? 'Recording captured. Play it back or delete to try again.'
            : 'Tap start and read the sentence clearly in a quiet room.'}
      </p>

      <div className="grid w-full max-w-md grid-cols-1 gap-2 sm:max-w-none sm:flex sm:flex-wrap sm:justify-center sm:gap-3">
        <PrimaryButton
          type="button"
          onClick={handleStartRecording}
          disabled={isRecording}
          className="sm:w-auto"
        >
          <Mic className="mr-2 h-4 w-4" />
          Start Recording
        </PrimaryButton>

        <PrimaryButton
          type="button"
          onClick={handleStopRecording}
          disabled={!isRecording}
          className="bg-brand-800 shadow-brand-200 hover:bg-brand-700 sm:w-auto dark:bg-brand-900 dark:hover:bg-brand-800"
        >
          <PauseCircle className="mr-2 h-4 w-4" />
          Stop Recording
        </PrimaryButton>

        <PrimaryButton
          type="button"
          onClick={handlePlayRecording}
          disabled={!hasRecording || isRecording}
          className="bg-brand-800 shadow-brand-200 hover:bg-brand-700 sm:w-auto dark:bg-brand-900 dark:hover:bg-brand-800"
        >
          <PlayCircle className="mr-2 h-4 w-4" />
          Play Recording
        </PrimaryButton>

        <PrimaryButton
          type="button"
          onClick={handleDeleteRecording}
          disabled={!hasRecording || isRecording}
          className="bg-brand-800 shadow-brand-200 hover:bg-brand-700 sm:w-auto dark:bg-brand-900 dark:hover:bg-brand-800"
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
