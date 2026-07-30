/**
 * Encode mono float32 PCM samples as a 16-bit PCM WAV blob.
 */
export function floatSamplesToWavBlob(samples, sampleRate) {
  if (!samples?.length) {
    throw new Error('No audio samples to encode')
  }
  if (!sampleRate || sampleRate < 8000) {
    throw new Error('Invalid sample rate')
  }

  const numChannels = 1
  const bytesPerSample = 2
  const blockAlign = numChannels * bytesPerSample
  const dataSize = samples.length * blockAlign
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  writeString(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeString(view, 8, 'WAVE')
  writeString(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * blockAlign, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, 16, true)
  writeString(view, 36, 'data')
  view.setUint32(40, dataSize, true)

  let offset = 44
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
    offset += 2
  }

  return new Blob([buffer], { type: 'audio/wav' })
}

/**
 * Convert browser MediaRecorder output (WebM/Opus) to PCM WAV via Web Audio.
 * Prefer floatSamplesToWavBlob from direct PCM capture when possible.
 */
export async function blobToWavBlob(blob) {
  if (!blob) {
    throw new Error('No audio blob to convert')
  }

  const header = new Uint8Array(await blob.slice(0, 12).arrayBuffer())
  const isRiffWav =
    header.length >= 12 &&
    String.fromCharCode(...header.slice(0, 4)) === 'RIFF' &&
    String.fromCharCode(...header.slice(8, 12)) === 'WAVE'

  if (isRiffWav) {
    return blob.type?.startsWith('audio/')
      ? blob
      : new Blob([blob], { type: 'audio/wav' })
  }

  const arrayBuffer = await blob.arrayBuffer()
  const AudioCtx = window.AudioContext || window.webkitAudioContext
  const audioContext = new AudioCtx()

  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0))
    const length = audioBuffer.length
    const mixed = new Float32Array(length)
    const channelCount = audioBuffer.numberOfChannels || 1

    for (let channel = 0; channel < channelCount; channel += 1) {
      const data = audioBuffer.getChannelData(channel)
      for (let i = 0; i < length; i += 1) {
        mixed[i] += data[i]
      }
    }
    for (let i = 0; i < length; i += 1) {
      mixed[i] /= channelCount
    }

    return floatSamplesToWavBlob(mixed, audioBuffer.sampleRate)
  } finally {
    await audioContext.close().catch(() => {})
  }
}

function writeString(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i))
  }
}
