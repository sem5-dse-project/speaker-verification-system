const fs = require('fs')
const path = require('path')
const crypto = require('crypto')
const multer = require('multer')

const toTimestampMs = () => {
  const now = new Date()
  const yyyy = now.getFullYear()
  const mm = String(now.getMonth() + 1).padStart(2, '0')
  const dd = String(now.getDate()).padStart(2, '0')
  const hh = String(now.getHours()).padStart(2, '0')
  const min = String(now.getMinutes()).padStart(2, '0')
  const ss = String(now.getSeconds()).padStart(2, '0')
  const ms = String(now.getMilliseconds()).padStart(3, '0')
  return `${yyyy}${mm}${dd}_${hh}${min}${ss}${ms}`
}

const shortId = () => crypto.randomBytes(3).toString('hex')

const sampleIndexFromName = (originalName) => {
  const match = String(originalName || '').match(/(?:enroll|verify)[_-]?(\d+)/i)
  return match ? match[1] : null
}

const buildUploader = (sampleType) => {
  const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      const baseDir = sampleType === 'enrollment' ? 'enrollments' : 'verifications'
      const userDir = path.join(__dirname, '..', 'uploads', baseDir, `user_${req.user.id}`)
      fs.mkdirSync(userDir, { recursive: true })
      cb(null, userDir)
    },
    filename: (req, file, cb) => {
      const prefix = sampleType === 'enrollment' ? 'enroll' : 'verify'
      const userId = req.user?.id ?? 'unknown'
      const sampleIdx = sampleIndexFromName(file.originalname)
      const samplePart = sampleIdx ? `s${sampleIdx}` : 's'
      // Example: enroll_u1_s2_20260730_083915142_a3f2c1.wav
      cb(
        null,
        `${prefix}_u${userId}_${samplePart}_${toTimestampMs()}_${shortId()}.wav`,
      )
    },
  })

  return multer({
    storage,
    limits: { fileSize: 20 * 1024 * 1024 },
    fileFilter: (_req, file, cb) => {
      const looksLikeWav =
        file.mimetype === 'audio/wav' ||
        file.mimetype === 'audio/x-wav' ||
        file.originalname.toLowerCase().endsWith('.wav')

      if (!looksLikeWav) {
        return cb(new Error('Only WAV audio files are allowed'))
      }

      return cb(null, true)
    },
  })
}

module.exports = {
  enrollmentUpload: buildUploader('enrollment'),
  verificationUpload: buildUploader('verification'),
  // Exported for unit tests
  sampleIndexFromName,
  toTimestampMs,
}
