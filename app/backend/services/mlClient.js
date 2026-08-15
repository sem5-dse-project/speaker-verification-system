const fs = require('fs')
const path = require('path')
const { Blob } = require('buffer')

const ML_SERVER_URL = (process.env.ML_SERVER_URL || 'http://localhost:8000').replace(/\/$/, '')

const isPcmWavFile = (absolutePath) => {
  try {
    const fd = fs.openSync(absolutePath, 'r')
    const header = Buffer.alloc(12)
    fs.readSync(fd, header, 0, 12, 0)
    fs.closeSync(fd)
    return (
      header.toString('ascii', 0, 4) === 'RIFF' &&
      header.toString('ascii', 8, 12) === 'WAVE'
    )
  } catch {
    return false
  }
}

const filePart = (absolutePath) => {
  const buffer = fs.readFileSync(absolutePath)
  // Node Blob needs a Uint8Array view for reliable multipart bytes
  const bytes = new Uint8Array(buffer)
  return new Blob([bytes], { type: 'audio/wav' })
}

const buildEnrollmentTemplate = async (absoluteFilePaths) => {
  if (!absoluteFilePaths?.length) {
    throw new Error('At least one enrollment audio file is required')
  }

  for (const filePath of absoluteFilePaths) {
    if (!isPcmWavFile(filePath)) {
      throw new Error(
        `Enrollment file is not a valid WAV (got non-RIFF audio, often old WebM): ${path.basename(filePath)}. Reset enrollment and record again.`,
      )
    }
  }

  const form = new FormData()
  for (const filePath of absoluteFilePaths) {
    form.append('files', filePart(filePath), path.basename(filePath))
  }

  const response = await fetch(`${ML_SERVER_URL}/enroll/template`, {
    method: 'POST',
    body: form,
  })

  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = data.detail || data.message || response.statusText
    throw new Error(`ML enroll/template failed: ${detail}`)
  }

  return data
}

const extractEmbedding = async (absoluteFilePath) => {
  if (!isPcmWavFile(absoluteFilePath)) {
    throw new Error('Audio file is not a valid WAV. Re-record and try again.')
  }

  const form = new FormData()
  form.append('files', filePart(absoluteFilePath), path.basename(absoluteFilePath))

  const response = await fetch(`${ML_SERVER_URL}/embed`, {
    method: 'POST',
    body: form,
  })

  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = data.detail || data.message || response.statusText
    throw new Error(`ML embed failed: ${detail}`)
  }

  const embedding = data.embeddings?.[0]
  if (!Array.isArray(embedding) || embedding.length === 0) {
    throw new Error('ML embed failed: missing embedding in response')
  }

  return {
    embedding,
    embedding_dim: Number(data.embedding_dim || embedding.length),
  }
}

const verifyAgainstTemplate = async (absoluteFilePath, embedding, threshold = null) => {
  if (!isPcmWavFile(absoluteFilePath)) {
    throw new Error(
      'Verification file is not a valid WAV. Re-record and try again.',
    )
  }

  const form = new FormData()
  form.append('file', filePart(absoluteFilePath), path.basename(absoluteFilePath))
  form.append('embedding', JSON.stringify(embedding))
  if (threshold !== null && threshold !== undefined) {
    form.append('threshold', String(threshold))
  }

  const response = await fetch(`${ML_SERVER_URL}/verify`, {
    method: 'POST',
    body: form,
  })

  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = data.detail || data.message || response.statusText
    throw new Error(`ML verify failed: ${detail}`)
  }

  return data
}

const REPLAY_DETECTION = String(process.env.REPLAY_DETECTION || 'true').toLowerCase() !== 'false'

const detectReplay = async (absoluteFilePath, threshold = null) => {
  if (!isPcmWavFile(absoluteFilePath)) {
    throw new Error(
      'Verification file is not a valid WAV. Re-record and try again.',
    )
  }

  const form = new FormData()
  form.append('file', filePart(absoluteFilePath), path.basename(absoluteFilePath))
  if (threshold !== null && threshold !== undefined) {
    form.append('threshold', String(threshold))
  }

  const response = await fetch(`${ML_SERVER_URL}/replay/detect`, {
    method: 'POST',
    body: form,
  })

  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = data.detail || data.message || response.statusText
    throw new Error(`ML replay/detect failed: ${detail}`)
  }

  return data
}

module.exports = {
  ML_SERVER_URL,
  REPLAY_DETECTION,
  isPcmWavFile,
  extractEmbedding,
  buildEnrollmentTemplate,
  verifyAgainstTemplate,
  detectReplay,
}
