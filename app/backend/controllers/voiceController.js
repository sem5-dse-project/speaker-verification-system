const fs = require('fs')
const path = require('path')
const voiceModel = require('../models/voiceModel')
const templateModel = require('../models/templateModel')
const mlClient = require('../services/mlClient')

const REQUIRED_ENROLLMENT_SAMPLES = Number(process.env.REQUIRED_ENROLLMENT_SAMPLES || 3)
const BACKEND_ROOT = path.join(__dirname, '..')

const toRelativePath = (absolutePath) =>
  path.relative(BACKEND_ROOT, absolutePath).replace(/\\/g, '/')

const toAbsolutePath = (relativePath) => path.join(BACKEND_ROOT, relativePath)

const resetEnrollment = async (req, res) => {
  try {
    const deleted = await voiceModel.deleteEnrollmentSamples(
      req.user.id,
      toAbsolutePath,
    )
    await templateModel.deleteTemplateByUserId(req.user.id)

    return res.json({
      success: true,
      message: 'Enrollment samples and template cleared',
      deleted_samples: deleted,
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to reset enrollment',
      error: error.message,
    })
  }
}

const uploadEnrollment = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'WAV file is required',
      })
    }

    if (!mlClient.isPcmWavFile(req.file.path)) {
      try {
        fs.unlinkSync(req.file.path)
      } catch {
        /* ignore */
      }
      return res.status(400).json({
        success: false,
        message:
          'Uploaded file is not a valid PCM WAV. Refresh the page and record again.',
      })
    }

    const relativePath = toRelativePath(req.file.path)
    const sample = await voiceModel.createVoiceSample(
      req.user.id,
      relativePath,
      'enrollment',
    )

    const enrollmentCount = await voiceModel.countEnrollmentSamples(req.user.id)
    let template = null
    let templateStatus = 'pending'

    if (enrollmentCount >= REQUIRED_ENROLLMENT_SAMPLES) {
      const latest = await voiceModel.getLatestEnrollmentSamples(
        req.user.id,
        REQUIRED_ENROLLMENT_SAMPLES,
      )
      const absolutePaths = latest.map((row) => toAbsolutePath(row.file_path))
      const mlResult = await mlClient.buildEnrollmentTemplate(absolutePaths)
      template = await templateModel.upsertTemplate(
        req.user.id,
        mlResult.embedding,
        mlResult.embedding_dim,
        mlResult.num_samples,
        null,
      )
      templateStatus = 'ready'
    }

    return res.status(201).json({
      success: true,
      message:
        templateStatus === 'ready'
          ? 'Enrollment audio uploaded and speaker template saved'
          : `Enrollment audio uploaded (${enrollmentCount}/${REQUIRED_ENROLLMENT_SAMPLES} samples)`,
      sample,
      enrollment_count: enrollmentCount,
      required_samples: REQUIRED_ENROLLMENT_SAMPLES,
      template_status: templateStatus,
      template: template
        ? {
            user_id: template.user_id,
            embedding_dim: template.embedding_dim,
            num_samples: template.num_samples,
            updated_at: template.updated_at,
          }
        : null,
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to upload enrollment audio',
      error: error.message,
    })
  }
}

const uploadVerification = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'WAV file is required',
      })
    }

    const relativePath = toRelativePath(req.file.path)
    const sample = await voiceModel.createVoiceSample(
      req.user.id,
      relativePath,
      'verification',
    )

    const template = await templateModel.getTemplateByUserId(req.user.id)
    if (!template) {
      return res.status(400).json({
        success: false,
        message:
          'No enrollment template found. Complete enrollment with 3 voice samples first.',
        sample,
      })
    }

    const mlResult = await mlClient.verifyAgainstTemplate(
      toAbsolutePath(relativePath),
      template.embedding,
      template.threshold,
    )

    return res.status(201).json({
      success: true,
      message: 'Verification complete',
      sample,
      result: {
        score: mlResult.score,
        threshold: mlResult.threshold,
        accepted: mlResult.accepted,
        decision: mlResult.decision,
      },
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to verify audio',
      error: error.message,
    })
  }
}

const getHistory = async (req, res) => {
  try {
    const history = await voiceModel.getVoiceHistoryByUserId(req.user.id)
    const template = await templateModel.getTemplateByUserId(req.user.id)
    return res.json({
      success: true,
      history,
      template: template
        ? {
            user_id: template.user_id,
            embedding_dim: template.embedding_dim,
            num_samples: template.num_samples,
            updated_at: template.updated_at,
            has_embedding: true,
          }
        : null,
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch voice history',
      error: error.message,
    })
  }
}

module.exports = {
  resetEnrollment,
  uploadEnrollment,
  uploadVerification,
  getHistory,
}
