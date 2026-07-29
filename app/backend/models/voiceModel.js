const { pool } = require('../config/db')

const createVoiceSample = async (userId, filePath, sampleType) => {
  const [result] = await pool.query(
    'INSERT INTO voice_samples (user_id, file_path, sample_type) VALUES (?, ?, ?)',
    [userId, filePath, sampleType],
  )

  return {
    id: result.insertId,
    user_id: userId,
    file_path: filePath,
    sample_type: sampleType,
  }
}

const getVoiceHistoryByUserId = async (userId) => {
  const [rows] = await pool.query(
    `
      SELECT id, user_id, file_path, sample_type, created_at
      FROM voice_samples
      WHERE user_id = ?
      ORDER BY created_at DESC
    `,
    [userId],
  )

  return rows
}

module.exports = {
  createVoiceSample,
  getVoiceHistoryByUserId,
}
