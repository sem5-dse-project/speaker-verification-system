const fs = require('fs')
const path = require('path')
const multer = require('multer')

const toTimestamp = () => {
  const now = new Date()
  const yyyy = now.getFullYear()
  const mm = String(now.getMonth() + 1).padStart(2, '0')
  const dd = String(now.getDate()).padStart(2, '0')
  const hh = String(now.getHours()).padStart(2, '0')
  const min = String(now.getMinutes()).padStart(2, '0')
  const ss = String(now.getSeconds()).padStart(2, '0')
  return `${yyyy}${mm}${dd}_${hh}${min}${ss}`
}

const buildUploader = (sampleType) => {
  const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      const baseDir = sampleType === 'enrollment' ? 'enrollments' : 'verifications'
      const userDir = path.join(__dirname, '..', 'uploads', baseDir, `user_${req.user.id}`)
      fs.mkdirSync(userDir, { recursive: true })
      cb(null, userDir)
    },
    filename: (_req, _file, cb) => {
      const prefix = sampleType === 'enrollment' ? 'enroll' : 'verify'
      cb(null, `${prefix}_${toTimestamp()}.wav`)
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
}
