const bcrypt = require('bcrypt')
const path = require('path')
const userModel = require('../models/userModel')
const collectionModel = require('../models/collectionModel')
const mlClient = require('../services/mlClient')

const toRelativePath = (absolutePath) => {
  const uploadsRoot = path.join(__dirname, '..', 'uploads')
  return path.relative(uploadsRoot, absolutePath).replace(/\\/g, '/')
}

const createAdmin = async (req, res) => {
  try {
    const { username, password } = req.body

    if (!username || !password) {
      return res.status(400).json({
        success: false,
        message: 'username and password are required',
      })
    }

    const trimmedUsername = String(username).trim()
    if (trimmedUsername.length < 3) {
      return res.status(400).json({
        success: false,
        message: 'Username must be at least 3 characters',
      })
    }

    if (String(password).length < 6) {
      return res.status(400).json({
        success: false,
        message: 'Password must be at least 6 characters',
      })
    }

    const existingUser = await userModel.findByUsername(trimmedUsername)
    if (existingUser) {
      return res.status(409).json({
        success: false,
        message: 'Username already exists',
      })
    }

    const passwordHash = await bcrypt.hash(password, 10)
    const admin = await userModel.createUser(trimmedUsername, passwordHash, 'admin')

    return res.status(201).json({
      success: true,
      message: 'Admin created successfully',
      admin: {
        id: admin.id,
        username: admin.username,
        role: admin.role,
      },
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to create admin',
      error: error.message,
    })
  }
}

const listAdmins = async (_req, res) => {
  try {
    const admins = await userModel.listAdmins()
    return res.json({
      success: true,
      admins,
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to list admins',
      error: error.message,
    })
  }
}

const uploadCollectionSample = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'WAV file is required',
      })
    }

    const speakerId = String(req.body.speaker_id || '').trim()
    const label = String(req.body.label || '').trim().toLowerCase()
    const consent = String(req.body.consent || '').toLowerCase() === 'true'

    if (!speakerId) {
      return res.status(400).json({
        success: false,
        message: 'speaker_id is required',
      })
    }

    if (!['live', 'replay'].includes(label)) {
      return res.status(400).json({
        success: false,
        message: 'label must be live or replay',
      })
    }

    if (!consent) {
      return res.status(400).json({
        success: false,
        message: 'Consent is required before uploading research audio',
      })
    }

    if (!mlClient.isPcmWavFile(req.file.path)) {
      return res.status(400).json({
        success: false,
        message: 'Uploaded file is not a valid PCM WAV',
      })
    }

    let replayScore = null
    let replayDecision = null

    if (mlClient.REPLAY_DETECTION) {
      try {
        const replay = await mlClient.detectReplay(req.file.path)
        replayScore = replay.score
        replayDecision = replay.decision
      } catch (error) {
        console.warn('Replay scoring failed for collection sample:', error.message)
      }
    }

    const relativePath = toRelativePath(req.file.path)
    const sample = await collectionModel.createSample({
      adminId: req.user.id,
      speakerId,
      label,
      filePath: relativePath,
      phrase: req.body.phrase,
      phoneModel: req.body.phone_model,
      distance: req.body.distance,
      volume: req.body.volume,
      notes: req.body.notes,
      consent,
      replayScore,
      replayDecision,
    })

    const counts = await collectionModel.countByLabel()

    return res.status(201).json({
      success: true,
      message: 'Collection sample saved',
      sample: {
        id: sample.id,
        speaker_id: speakerId,
        label,
        file_path: relativePath,
        replay_score: replayScore,
        replay_decision: replayDecision,
      },
      counts,
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to save collection sample',
      error: error.message,
    })
  }
}

const listCollectionSamples = async (_req, res) => {
  try {
    const samples = await collectionModel.listSamples()
    const counts = await collectionModel.countByLabel()

    return res.json({
      success: true,
      samples,
      counts,
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to list collection samples',
      error: error.message,
    })
  }
}

const exportCollectionMetadata = async (_req, res) => {
  try {
    const samples = await collectionModel.listSamples()
    const header = [
      'id',
      'speaker_id',
      'label',
      'file_path',
      'phrase',
      'phone_model',
      'distance',
      'volume',
      'notes',
      'replay_score',
      'replay_decision',
      'collected_by',
      'created_at',
    ]

    const escapeCsv = (value) => {
      const text = value == null ? '' : String(value)
      if (text.includes(',') || text.includes('"') || text.includes('\n')) {
        return `"${text.replace(/"/g, '""')}"`
      }
      return text
    }

    const rows = samples.map((sample) =>
      [
        sample.id,
        sample.speaker_id,
        sample.label,
        sample.file_path,
        sample.phrase,
        sample.phone_model,
        sample.distance,
        sample.volume,
        sample.notes,
        sample.replay_score,
        sample.replay_decision,
        sample.collected_by,
        sample.created_at,
      ]
        .map(escapeCsv)
        .join(','),
    )

    const csv = [header.join(','), ...rows].join('\n')

    res.setHeader('Content-Type', 'text/csv; charset=utf-8')
    res.setHeader(
      'Content-Disposition',
      'attachment; filename="collection_metadata.csv"',
    )
    return res.send(csv)
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to export collection metadata',
      error: error.message,
    })
  }
}

module.exports = {
  createAdmin,
  listAdmins,
  uploadCollectionSample,
  listCollectionSamples,
  exportCollectionMetadata,
}
