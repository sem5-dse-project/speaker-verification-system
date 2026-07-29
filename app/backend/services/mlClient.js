const fs = require('fs')
const path = require('path')
const { Blob } = require('buffer')

const ML_SERVER_URL = (process.env.ML_SERVER_URL || 'http://localhost:8000').replace(/\/$/, '')

const buildEnrollmentTemplate = async (absoluteFilePaths) => {
  if (!absoluteFilePaths?.length) {
    throw new Error('At least one enrollment audio file is required')
  }

  const form = new FormData()
  for (const filePath of absoluteFilePaths) {
    const buffer = fs.readFileSync(filePath)
    const blob = new Blob([buffer], { type: 'audio/wav' })
    form.append('files', blob, path.basename(filePath))
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

const verifyAgainstTemplate = async (absoluteFilePath, embedding, threshold = null) => {
  const buffer = fs.readFileSync(absoluteFilePath)
  const form = new FormData()
  form.append('file', new Blob([buffer], { type: 'audio/wav' }), path.basename(absoluteFilePath))
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

module.exports = {
  ML_SERVER_URL,
  buildEnrollmentTemplate,
  verifyAgainstTemplate,
}
