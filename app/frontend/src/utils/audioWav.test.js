import { describe, it, expect } from 'vitest'
import { floatSamplesToWavBlob, blobToWavBlob } from '../utils/audioWav.js'

describe('floatSamplesToWavBlob', () => {
  it('encodes a RIFF/WAVE PCM blob', async () => {
    const samples = new Float32Array([0, 0.5, -0.5, 1, -1])
    const blob = floatSamplesToWavBlob(samples, 16000)

    expect(blob.type).toBe('audio/wav')
    expect(blob.size).toBe(44 + samples.length * 2)

    const header = new Uint8Array(await blob.slice(0, 12).arrayBuffer())
    expect(String.fromCharCode(...header.slice(0, 4))).toBe('RIFF')
    expect(String.fromCharCode(...header.slice(8, 12))).toBe('WAVE')
  })

  it('rejects empty samples', () => {
    expect(() => floatSamplesToWavBlob(new Float32Array(0), 16000)).toThrow(
      'No audio samples to encode',
    )
  })

  it('rejects invalid sample rate', () => {
    expect(() => floatSamplesToWavBlob(new Float32Array([0.1]), 100)).toThrow(
      'Invalid sample rate',
    )
  })
})

describe('blobToWavBlob', () => {
  it('passes through existing WAV blobs', async () => {
    const wav = floatSamplesToWavBlob(new Float32Array([0.1, -0.1]), 16000)
    const result = await blobToWavBlob(wav)
    expect(result).toBe(wav)
  })

  it('throws when blob is missing', async () => {
    await expect(blobToWavBlob(null)).rejects.toThrow('No audio blob to convert')
  })
})
