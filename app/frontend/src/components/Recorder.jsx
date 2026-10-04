import { useEffect, useMemo, useRef, useState } from 'react'
import { Mic, PauseCircle, PlayCircle, RefreshCcw, Trash2 } from 'lucide-react'
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

export const TARGET_RECORDING_SECONDS = 4
const MIN_RECORDING_SECONDS = 0.5

export function reachedRecordingLimit(
  totalSamples,
  sampleRate,
  targetSeconds = TARGET_RECORDING_SECONDS,
) {
  if (!sampleRate || sampleRate <= 0) {
    return false
  }
  return totalSamples >= sampleRate * targetSeconds
}

function idleHint(readsSentence) {
  if (readsSentence) {
    return 'Read the sentence aloud for about 4 seconds in a quiet room. Recording stops automatically.'
  }
  return 'Speak for about 4 seconds in a quiet room. Recording stops automatically.'
}

function Recorder({ onRecordingChange, onRecorderError, readsSentence = false }) {
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
  const stoppingRef = useRef(false)
  const stopRef = useRef(() => {})

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
      stoppingRef.current = false
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

        const totalSamples = pcmChunksRef.current.reduce((total, chunk) => total + chunk.length, 0)
        if (reachedRecordingLimit(totalSamples, sampleRateRef.current)) {
          recordingRef.current = false
          window.setTimeout(() => stopRef.current(), 0)
        }
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
    if (stoppingRef.current) {
      return
    }

    const chunks = pcmChunksRef.current
    if (!recordingRef.current && chunks.length === 0) {
      return
    }

    stoppingRef.current = true
    stopTimer()
    recordingRef.current = false

    try {
      const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0)

      if (totalLength < sampleRateRef.current * MIN_RECORDING_SECONDS) {
        pcmChunksRef.current = []
        stoppingRef.current = false
        cleanupAudioGraph()
        setStatus('ready')
        setLevels(emptyLevels())
        onRecorderError('Recording too short. Please speak for at least half a second.')
        return
      }

      setSeconds(
        Math.min(
          TARGET_RECORDING_SECONDS,
          Math.max(1, Math.round(totalLength / sampleRateRef.current)),
        ),
      )

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
      stoppingRef.current = false
    }
  }

  stopRef.current = handleStopRecording

  const handlePlayRecording = async () => {
    if (audioRef.current && hasRecording) {
      await audioRef.current.play()
    }
  }

  const handleDeleteRecording = () => {
    stopTimer()
    stoppingRef.current = false
    pcmChunksRef.current = []
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
        {isRecording
          ? `${formatSeconds(seconds)} / ${formatSeconds(TARGET_RECORDING_SECONDS)}`
          : formatSeconds(seconds)}
      </p>

      <p className="max-w-sm text-xs text-subtle sm:text-sm">
        {isRecording
          ? 'Speak naturally. Recording stops at 4 seconds.'
          : hasRecording
            ? 'Recording captured. Play it back, or record again.'
            : idleHint(readsSentence)}
      </p>

      <div className="grid w-full max-w-md grid-cols-1 gap-2 sm:max-w-none sm:flex sm:flex-wrap sm:justify-center sm:gap-3">
        {!isRecording && !hasRecording && (
          <PrimaryButton type="button" onClick={handleStartRecording} className="sm:w-auto">
            <Mic className="mr-2 h-4 w-4" />
            Start Recording
          </PrimaryButton>
        )}

        {isRecording && (
          <PrimaryButton type="button" onClick={handleStopRecording} className="sm:w-auto">
            <PauseCircle className="mr-2 h-4 w-4" />
            Stop Recording
          </PrimaryButton>
        )}

        {hasRecording && !isRecording && (
          <>
            <button type="button" onClick={handlePlayRecording} className="btn-secondary">
              <PlayCircle className="h-4 w-4" />
              Play Recording
            </button>
            <button type="button" onClick={handleDeleteRecording} className="btn-secondary">
              <Trash2 className="h-4 w-4" />
              Delete Recording
            </button>
            <button type="button" onClick={handleStartRecording} className="btn-secondary">
              <RefreshCcw className="h-4 w-4" />
              Record again
            </button>
          </>
        )}
      </div>

      <audio ref={audioRef} src={audioUrl ?? undefined} className="hidden" />
    </section>
  )
}

export default Recorder
