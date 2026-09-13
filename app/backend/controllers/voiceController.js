const fs = require('fs')
const path = require('path')
const bcrypt = require('bcrypt')
const jwt = require('jsonwebtoken')
const voiceModel = require('../models/voiceModel')
const templateModel = require('../models/templateModel')
const userModel = require('../models/userModel')
const verificationLogModel = require('../models/verificationLogModel')
const mlClient = require('../services/mlClient')
const {
  findBestTemplateMatch,
  cosineSimilarity,
  decideByThreshold,
  passesIdentifyGate,
} = require('../services/similarity')
const {
  createVoiceLoginSession,
  getVoiceLoginSession,
  deleteVoiceLoginSession,
} = require('../services/voiceLoginCache')

const REQUIRED_ENROLLMENT_SAMPLES = Number(process.env.REQUIRED_ENROLLMENT_SAMPLES || 3)
const DEFAULT_VERIFY_THRESHOLD = Number(process.env.DEFAULT_VERIFY_THRESHOLD || 0.25)
/** Min cosine for open-set identify (unenrolled speakers must not get a userid). */
const IDENTIFY_THRESHOLD = Number(
  process.env.IDENTIFY_THRESHOLD || DEFAULT_VERIFY_THRESHOLD,
)
/** Min gap between best and second-best scores; 0 disables the margin check. */
const IDENTIFY_MARGIN = Number(process.env.IDENTIFY_MARGIN || 0.05)
const BACKEND_ROOT = path.join(__dirname, '..')

const toRelativePath = (absolutePath) =>
  path.relative(BACKEND_ROOT, absolutePath).replace(/\\/g, '/')

const toAbsolutePath = (relativePath) => path.join(BACKEND_ROOT, relativePath)

const signUserToken = (user) =>
  jwt.sign(
    { id: user.id, username: user.username },
    process.env.JWT_SECRET,
    { expiresIn: '1d' },
  )

const runReplayDetection = async (absolutePath) => {
  if (!mlClient.REPLAY_DETECTION) {
    return null
  }

  try {
    return await mlClient.detectReplay(absolutePath)
  } catch (error) {
    const message = error.message || ''
    if (message.includes('Replay checkpoint not found')) {
      const unavailable = new Error('Replay detection is unavailable on this server')
      unavailable.statusCode = 503
      throw unavailable
    }
    throw error
  }
}

const identifyVoice = async (req, res) => {
  let absolutePath = null
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'WAV file is required',
      })
    }

    const relativePath = toRelativePath(req.file.path)
    absolutePath = toAbsolutePath(relativePath)

    const replay = await runReplayDetection(absolutePath)
    if (replay?.decision === 'NO_SPEECH') {
      return res.status(400).json({
        success: false,
        message: 'No speech detected - please speak clearly and try again',
        replay,
      })
    }

    if (replay?.is_replay) {
      return res.status(403).json({
        success: false,
        message: 'Voice login rejected: replay attack detected',
        replay,
      })
    }

    const { embedding, embedding_dim } = await mlClient.extractEmbedding(absolutePath)
    const templates = await templateModel.getAllTemplatesWithUsers()

    if (!templates.length) {
      return res.status(400).json({
        success: false,
        message: 'No enrolled voice templates are available for identification',
      })
    }

    const bestMatch = findBestTemplateMatch(embedding, templates)
    const identifyGate = passesIdentifyGate(bestMatch, {
      threshold: IDENTIFY_THRESHOLD,
      margin: IDENTIFY_MARGIN,
    })

    if (!identifyGate.accepted) {
      const message =
        identifyGate.reason === 'ambiguous'
          ? 'Voice match is ambiguous between users. Please use password login or re-record clearly.'
          : 'Could not identify a matching enrolled user. Enroll your voice first, or use password login.'

      return res.status(401).json({
        success: false,
        message,
        identify: {
          reason: identifyGate.reason,
          score: identifyGate.score,
          second_score: identifyGate.second_score,
          margin: identifyGate.margin,
          threshold: identifyGate.threshold,
          required_margin: identifyGate.required_margin,
        },
      })
    }

    const cached = createVoiceLoginSession({
      probe_embedding: embedding,
      embedding_dim,
      identified_user_id: bestMatch.user_id,
      identified_username: bestMatch.username,
      identify_score: bestMatch.score,
      replay,
    })

    return res.json({
      success: true,
      message: 'Voice identification complete',
      temporary_login_token: cached.token,
      expires_at: cached.expires_at,
      ttl_seconds: cached.ttl_seconds,
      identified_user: {
        id: bestMatch.user_id,
        username: bestMatch.username,
      },
      similarity_score: bestMatch.score,
      identify: {
        score: identifyGate.score,
        second_score: identifyGate.second_score,
        margin: identifyGate.margin,
        threshold: identifyGate.threshold,
        required_margin: identifyGate.required_margin,
      },
    })
  } catch (error) {
    if (error.statusCode === 503) {
      return res.status(503).json({
        success: false,
        message: error.message,
      })
    }

    return res.status(500).json({
      success: false,
      message: 'Failed to identify voice',
      error: error.message,
    })
  } finally {
    if (absolutePath) {
      try {
        fs.unlinkSync(absolutePath)
      } catch {
        /* ignore cleanup errors */
      }
    }
  }
}

const loginWithVoice = async (req, res) => {
  try {
    const { temporary_login_token, password } = req.body

    if (!temporary_login_token || !password) {
      return res.status(400).json({
        success: false,
        message: 'temporary_login_token and password are required',
      })
    }

    const session = getVoiceLoginSession(temporary_login_token)
    if (!session) {
      return res.status(401).json({
        success: false,
        message: 'Voice login session is missing or expired. Please record again.',
      })
    }

    const user = await userModel.findAuthById(session.identified_user_id)
    if (!user) {
      deleteVoiceLoginSession(temporary_login_token)
      return res.status(404).json({
        success: false,
        message: 'Identified user no longer exists',
      })
    }

    const validPassword = await bcrypt.compare(password, user.password)
    if (!validPassword) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials',
      })
    }

    const template = await templateModel.getTemplateByUserId(user.id)
    if (!template) {
      deleteVoiceLoginSession(temporary_login_token)
      return res.status(400).json({
        success: false,
        message: 'No enrollment template found for identified user',
      })
    }

    const score = cosineSimilarity(session.probe_embedding, template.embedding)
    const threshold =
      template.threshold === null || template.threshold === undefined
        ? DEFAULT_VERIFY_THRESHOLD
        : Number(template.threshold)
    const voiceDecision = decideByThreshold(score, threshold)

    const log = await verificationLogModel.createVerificationLog({
      userId: user.id,
      voiceSampleId: null,
      score: voiceDecision.score,
      threshold: voiceDecision.threshold,
      accepted: voiceDecision.accepted,
      decision: voiceDecision.decision,
    })

    deleteVoiceLoginSession(temporary_login_token)

    if (!voiceDecision.accepted) {
      return res.status(401).json({
        success: false,
        message: 'Voice verification failed for identified user',
        identified_user: {
          id: user.id,
          username: user.username,
        },
        voice: voiceDecision,
        log,
      })
    }

    const token = signUserToken(user)

    return res.json({
      success: true,
      token,
      user: {
        id: user.id,
        username: user.username,
      },
      voice: voiceDecision,
      log,
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to complete voice login',
      error: error.message,
    })
  }
}

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

    const absolutePath = toAbsolutePath(relativePath)
    const replay = await runReplayDetection(absolutePath)

    if (replay) {
      if (replay.decision === 'NO_SPEECH') {
        const log = await verificationLogModel.createVerificationLog({
          userId: req.user.id,
          voiceSampleId: sample.id,
          score: replay.score,
          threshold: replay.threshold,
          accepted: false,
          decision: 'NO_SPEECH',
        })

        return res.status(201).json({
          success: true,
          message: 'No speech detected - please speak clearly and try again',
          sample,
          replay,
          result: {
            score: replay.score,
            threshold: replay.threshold,
            threshold_low: replay.threshold_low,
            threshold_high: replay.threshold_high,
            accepted: false,
            decision: 'NO_SPEECH',
            rms: replay.rms,
            replay: replay.replay,
            la: replay.la,
          },
          log,
        })
      }

      if (replay.decision === 'REPLAY' || replay.is_replay) {
        const log = await verificationLogModel.createVerificationLog({
          userId: req.user.id,
          voiceSampleId: sample.id,
          score: replay.score,
          threshold: replay.threshold,
          accepted: false,
          decision: 'REPLAY',
        })

        return res.status(201).json({
          success: true,
          message: 'Verification rejected: replay attack detected',
          sample,
          replay,
          result: {
            score: replay.score,
            threshold: replay.threshold,
            threshold_low: replay.threshold_low,
            threshold_high: replay.threshold_high,
            accepted: false,
            decision: 'REPLAY',
            is_synthetic: false,
            replay: replay.replay,
            la: replay.la,
          },
          log,
        })
      }

      if (replay.decision === 'SYNTHETIC' || replay.is_synthetic) {
        const log = await verificationLogModel.createVerificationLog({
          userId: req.user.id,
          voiceSampleId: sample.id,
          score: replay.score,
          threshold: replay.threshold,
          accepted: false,
          decision: 'SYNTHETIC',
        })

        return res.status(201).json({
          success: true,
          message: 'Verification rejected: synthetic spoof detected',
          sample,
          replay,
          result: {
            score: replay.score,
            threshold: replay.threshold,
            threshold_low: replay.threshold_low,
            threshold_high: replay.threshold_high,
            accepted: false,
            decision: 'SYNTHETIC',
            is_synthetic: true,
            replay: replay.replay,
            la: replay.la,
          },
          log,
        })
      }

      if (replay.decision === 'UNCERTAIN') {
        const log = await verificationLogModel.createVerificationLog({
          userId: req.user.id,
          voiceSampleId: sample.id,
          score: replay.score,
          threshold: replay.threshold,
          accepted: false,
          decision: 'UNCERTAIN',
        })

        return res.status(201).json({
          success: true,
          message: 'Audio quality uncertain - please re-record and try again',
          sample,
          replay,
          result: {
            score: replay.score,
            threshold: replay.threshold,
            threshold_low: replay.threshold_low,
            threshold_high: replay.threshold_high,
            accepted: false,
            decision: 'UNCERTAIN',
            replay: replay.replay,
            la: replay.la,
          },
          log,
        })
      }
    }

    const mlResult = await mlClient.verifyAgainstTemplate(
      absolutePath,
      template.embedding,
      null, // use ML server DEFAULT_THRESHOLD (env); avoid stale per-user thr
    )

    const log = await verificationLogModel.createVerificationLog({
      userId: req.user.id,
      voiceSampleId: sample.id,
      score: mlResult.score,
      threshold: mlResult.threshold,
      accepted: mlResult.accepted,
      decision: mlResult.decision,
    })

    return res.status(201).json({
      success: true,
      message: 'Verification complete',
      sample,
      replay,
      result: {
        score: mlResult.score,
        threshold: mlResult.threshold,
        accepted: mlResult.accepted,
        decision: mlResult.decision,
        replay: replay?.replay,
        la: replay?.la,
      },
      log,
    })
  } catch (error) {
    if (error.statusCode === 503) {
      return res.status(503).json({
        success: false,
        message: error.message,
      })
    }

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
    const verification_logs = await verificationLogModel.getVerificationLogsByUserId(
      req.user.id,
    )
    return res.json({
      success: true,
      history,
      verification_logs,
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

const getVerificationLogs = async (req, res) => {
  try {
    const limit = Number(req.query.limit) || 50
    const logs = await verificationLogModel.getVerificationLogsByUserId(
      req.user.id,
      limit,
    )
    return res.json({
      success: true,
      logs,
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch verification logs',
      error: error.message,
    })
  }
}

module.exports = {
  identifyVoice,
  loginWithVoice,
  resetEnrollment,
  uploadEnrollment,
  uploadVerification,
  getHistory,
  getVerificationLogs,
}
