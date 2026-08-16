const express = require('express')
const voiceController = require('../controllers/voiceController')
const authMiddleware = require('../middleware/authMiddleware')
const { enrollmentUpload, verificationUpload } = require('../middleware/upload')

const router = express.Router()

router.post('/identify', verificationUpload.single('audio'), voiceController.identifyVoice)
router.post('/login', voiceController.loginWithVoice)

router.post('/enroll/reset', authMiddleware, voiceController.resetEnrollment)
router.post('/enroll', authMiddleware, enrollmentUpload.single('audio'), voiceController.uploadEnrollment)
router.post('/verify', authMiddleware, verificationUpload.single('audio'), voiceController.uploadVerification)
router.get('/verification-logs', authMiddleware, voiceController.getVerificationLogs)
router.get('/history', authMiddleware, voiceController.getHistory)

module.exports = router
