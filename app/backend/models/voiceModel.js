const fs = require('fs')
const { pool } = require('../config/db')

const createVoiceSample = async (userId, filePath, sampleType) => {
  const { rows } = await pool.query(
    `INSERT INTO voice_samples (user_id, file_path, sample_type)
     VALUES ($1, $2, $3)
     RETURNING id, user_id, file_path, sample_type`,
    [userId, filePath, sampleType],
  )

  return rows[0]
}

const getVoiceHistoryByUserId = async (userId) => {
  const { rows } = await pool.query(
    `
      SELECT id, user_id, file_path, sample_type, created_at
      FROM voice_samples
      WHERE user_id = $1
      ORDER BY created_at DESC
    `,
    [userId],
  )

  return rows
}

const countEnrollmentSamples = async (userId) => {
  const { rows } = await pool.query(
    `
      SELECT COUNT(*) AS count
      FROM voice_samples
      WHERE user_id = $1 AND sample_type = 'enrollment'
    `,
    [userId],
  )
  return Number(rows[0].count)
}

const getLatestEnrollmentSamples = async (userId, limit = 3) => {
  const { rows } = await pool.query(
    `
      SELECT id, user_id, file_path, sample_type, created_at
      FROM voice_samples
      WHERE user_id = $1 AND sample_type = 'enrollment'
      ORDER BY created_at DESC
      LIMIT $2
    `,
    [userId, limit],
  )
  // oldest → newest for stable template averaging order
  return rows.reverse()
}

const listEnrollmentSamples = async (userId) => {
  const { rows } = await pool.query(
    `
      SELECT id, user_id, file_path, sample_type, created_at
      FROM voice_samples
      WHERE user_id = $1 AND sample_type = 'enrollment'
      ORDER BY created_at ASC
    `,
    [userId],
  )
  return rows
}

const deleteEnrollmentSamples = async (userId, resolveAbsolutePath) => {
  const rows = await listEnrollmentSamples(userId)

  for (const row of rows) {
    const absolutePath = resolveAbsolutePath(row.file_path)
    try {
      if (fs.existsSync(absolutePath)) {
        fs.unlinkSync(absolutePath)
      }
    } catch {
      /* ignore missing/locked files */
    }
  }

  await pool.query(
    `
      DELETE FROM voice_samples
      WHERE user_id = $1 AND sample_type = 'enrollment'
    `,
    [userId],
  )

  return rows.length
}

module.exports = {
  createVoiceSample,
  getVoiceHistoryByUserId,
  countEnrollmentSamples,
  getLatestEnrollmentSamples,
  listEnrollmentSamples,
  deleteEnrollmentSamples,
}
