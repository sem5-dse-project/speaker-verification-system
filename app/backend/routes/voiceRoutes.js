const express = require('express')
const voiceController = require('../controllers/voiceController')
const authMiddleware = require('../middleware/authMiddleware')
const { enrollmentUpload, verificationUpload } = require('../middleware/upload')

const router = express.Router()

router.post('/enroll/reset', authMiddleware, voiceController.resetEnrollment)
router.post('/enroll', authMiddleware, enrollmentUpload.single('audio'), voiceController.uploadEnrollment)
router.post('/verify', authMiddleware, verificationUpload.single('audio'), voiceController.uploadVerification)
router.get('/history', authMiddleware, voiceController.getHistory)

module.exports = router
