jest.mock('../models/voiceModel')
jest.mock('../models/templateModel')
jest.mock('../models/verificationLogModel')
jest.mock('../services/mlClient')

const voiceModel = require('../models/voiceModel')
const templateModel = require('../models/templateModel')
const verificationLogModel = require('../models/verificationLogModel')
const mlClient = require('../services/mlClient')
const {
  uploadVerification,
  resetEnrollment,
  getVerificationLogs,
} = require('../controllers/voiceController')
const { mockRes } = require('./helpers')

describe('voiceController', () => {
  beforeEach(() => {
    mlClient.REPLAY_DETECTION = true
    mlClient.detectReplay.mockResolvedValue({
      score: 0.1,
      threshold: 0.76,
      is_replay: false,
      accepted: true,
      decision: 'LIVE',
    })
  })

  describe('uploadVerification', () => {
    it('returns 400 when no file is uploaded', async () => {
      const req = { file: null, user: { id: 1 } }
      const res = mockRes()

      await uploadVerification(req, res)

      expect(res.status).toHaveBeenCalledWith(400)
      expect(voiceModel.createVoiceSample).not.toHaveBeenCalled()
    })

    it('returns 400 when enrollment template is missing', async () => {
      voiceModel.createVoiceSample.mockResolvedValue({
        id: 10,
        user_id: 1,
        file_path: 'uploads/verifications/user_1/verify.wav',
        sample_type: 'verification',
      })
      templateModel.getTemplateByUserId.mockResolvedValue(null)

      const req = {
        user: { id: 1 },
        file: { path: 'D:\\tmp\\verify.wav' },
      }
      const res = mockRes()

      await uploadVerification(req, res)

      expect(res.status).toHaveBeenCalledWith(400)
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining('No enrollment template'),
        }),
      )
      expect(mlClient.verifyAgainstTemplate).not.toHaveBeenCalled()
    })

    it('rejects replay before speaker verify', async () => {
      voiceModel.createVoiceSample.mockResolvedValue({
        id: 10,
        user_id: 1,
        file_path: 'uploads/verifications/user_1/verify.wav',
        sample_type: 'verification',
      })
      templateModel.getTemplateByUserId.mockResolvedValue({
        user_id: 1,
        embedding: [0.1, 0.2],
        threshold: 0.25,
      })
      mlClient.detectReplay.mockResolvedValue({
        score: 0.91,
        threshold: 0.74,
        threshold_low: 0.64,
        threshold_high: 0.84,
        is_replay: true,
        accepted: false,
        decision: 'REPLAY',
      })
      verificationLogModel.createVerificationLog.mockResolvedValue({
        id: 5,
        decision: 'REPLAY',
      })

      const req = { user: { id: 1 }, file: { path: pathJoinSafe() } }
      const res = mockRes()

      await uploadVerification(req, res)

      expect(mlClient.verifyAgainstTemplate).not.toHaveBeenCalled()
      expect(verificationLogModel.createVerificationLog).toHaveBeenCalledWith(
        expect.objectContaining({ decision: 'REPLAY', accepted: false }),
      )
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          result: expect.objectContaining({ decision: 'REPLAY' }),
        }),
      )
    })

    it('asks re-record on no speech', async () => {
      voiceModel.createVoiceSample.mockResolvedValue({
        id: 10,
        user_id: 1,
        file_path: 'uploads/verifications/user_1/verify.wav',
        sample_type: 'verification',
      })
      templateModel.getTemplateByUserId.mockResolvedValue({
        user_id: 1,
        embedding: [0.1, 0.2],
        threshold: 0.45,
      })
      mlClient.detectReplay.mockResolvedValue({
        score: 0,
        threshold: 0.15,
        threshold_low: 0.05,
        threshold_high: 0.25,
        is_replay: false,
        accepted: false,
        decision: 'NO_SPEECH',
        rms: 0.001,
      })
      verificationLogModel.createVerificationLog.mockResolvedValue({
        id: 7,
        decision: 'NO_SPEECH',
      })

      const req = { user: { id: 1 }, file: { path: pathJoinSafe() } }
      const res = mockRes()

      await uploadVerification(req, res)

      expect(mlClient.verifyAgainstTemplate).not.toHaveBeenCalled()
      expect(verificationLogModel.createVerificationLog).toHaveBeenCalledWith(
        expect.objectContaining({ decision: 'NO_SPEECH', accepted: false }),
      )
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining('No speech'),
          result: expect.objectContaining({ decision: 'NO_SPEECH' }),
        }),
      )
    })

    it('asks re-record on uncertain replay band', async () => {
      voiceModel.createVoiceSample.mockResolvedValue({
        id: 10,
        user_id: 1,
        file_path: 'uploads/verifications/user_1/verify.wav',
        sample_type: 'verification',
      })
      templateModel.getTemplateByUserId.mockResolvedValue({
        user_id: 1,
        embedding: [0.1, 0.2],
        threshold: 0.25,
      })
      mlClient.detectReplay.mockResolvedValue({
        score: 0.72,
        threshold: 0.74,
        threshold_low: 0.64,
        threshold_high: 0.84,
        is_replay: false,
        accepted: false,
        decision: 'UNCERTAIN',
      })
      verificationLogModel.createVerificationLog.mockResolvedValue({
        id: 6,
        decision: 'UNCERTAIN',
      })

      const req = { user: { id: 1 }, file: { path: pathJoinSafe() } }
      const res = mockRes()

      await uploadVerification(req, res)

      expect(mlClient.verifyAgainstTemplate).not.toHaveBeenCalled()
      expect(verificationLogModel.createVerificationLog).toHaveBeenCalledWith(
        expect.objectContaining({ decision: 'UNCERTAIN', accepted: false }),
      )
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining('re-record'),
          result: expect.objectContaining({ decision: 'UNCERTAIN' }),
        }),
      )
    })

    it('scores audio, saves log, and returns result', async () => {
      const sample = {
        id: 10,
        user_id: 1,
        file_path: 'uploads/verifications/user_1/verify.wav',
        sample_type: 'verification',
      }
      voiceModel.createVoiceSample.mockResolvedValue(sample)
      templateModel.getTemplateByUserId.mockResolvedValue({
        user_id: 1,
        embedding: [0.1, 0.2],
        threshold: 0.25,
      })
      mlClient.verifyAgainstTemplate.mockResolvedValue({
        score: 0.8,
        threshold: 0.25,
        accepted: true,
        decision: 'ACCEPT',
      })
      const log = {
        id: 5,
        user_id: 1,
        voice_sample_id: 10,
        score: 0.8,
        threshold: 0.25,
        accepted: true,
        decision: 'ACCEPT',
      }
      verificationLogModel.createVerificationLog.mockResolvedValue(log)

      const req = {
        user: { id: 1 },
        file: { path: pathJoinSafe() },
      }
      const res = mockRes()

      await uploadVerification(req, res)

      expect(mlClient.detectReplay).toHaveBeenCalled()
      expect(verificationLogModel.createVerificationLog).toHaveBeenCalledWith(
        expect.objectContaining({
          userId: 1,
          voiceSampleId: 10,
          score: 0.8,
          decision: 'ACCEPT',
        }),
      )
      expect(res.status).toHaveBeenCalledWith(201)
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: true,
          result: expect.objectContaining({ decision: 'ACCEPT' }),
          log,
        }),
      )
    })
  })

  describe('resetEnrollment', () => {
    it('clears samples and template', async () => {
      voiceModel.deleteEnrollmentSamples.mockResolvedValue(3)
      templateModel.deleteTemplateByUserId.mockResolvedValue(true)

      const req = { user: { id: 1 } }
      const res = mockRes()

      await resetEnrollment(req, res)

      expect(voiceModel.deleteEnrollmentSamples).toHaveBeenCalled()
      expect(templateModel.deleteTemplateByUserId).toHaveBeenCalledWith(1)
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: true,
          deleted_samples: 3,
        }),
      )
    })
  })

  describe('getVerificationLogs', () => {
    it('returns logs for the current user', async () => {
      const logs = [{ id: 1, decision: 'ACCEPT' }]
      verificationLogModel.getVerificationLogsByUserId.mockResolvedValue(logs)

      const req = { user: { id: 1 }, query: {} }
      const res = mockRes()

      await getVerificationLogs(req, res)

      expect(verificationLogModel.getVerificationLogsByUserId).toHaveBeenCalledWith(
        1,
        50,
      )
      expect(res.json).toHaveBeenCalledWith({ success: true, logs })
    })
  })
})

function pathJoinSafe() {
  // Absolute-looking path so toAbsolutePath/path.relative still work in controller
  return require('path').join(__dirname, 'fixtures', 'verify.wav')
}
