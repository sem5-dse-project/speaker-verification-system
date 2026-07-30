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
