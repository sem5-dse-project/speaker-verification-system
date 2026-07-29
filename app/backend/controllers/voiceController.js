const path = require('path')
const voiceModel = require('../models/voiceModel')

const toRelativePath = (absolutePath) => {
  const backendRoot = path.join(__dirname, '..')
  return path.relative(backendRoot, absolutePath).replace(/\\/g, '/')
}

const uploadEnrollment = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'WAV file is required',
      })
    }

    const relativePath = toRelativePath(req.file.path)
    const sample = await voiceModel.createVoiceSample(req.user.id, relativePath, 'enrollment')

    return res.status(201).json({
      success: true,
      message: 'Enrollment audio uploaded successfully',
      sample,
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
    const sample = await voiceModel.createVoiceSample(req.user.id, relativePath, 'verification')

    return res.status(201).json({
      success: true,
      message: 'Audio uploaded successfully',
      result: 'Verification logic not implemented yet',
      sample,
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to upload verification audio',
      error: error.message,
    })
  }
}

const getHistory = async (req, res) => {
  try {
    const history = await voiceModel.getVoiceHistoryByUserId(req.user.id)
    return res.json({
      success: true,
      history,
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
  uploadEnrollment,
  uploadVerification,
  getHistory,
}
