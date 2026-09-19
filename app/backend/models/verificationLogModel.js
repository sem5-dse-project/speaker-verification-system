const { pool } = require('../config/db')

const createVerificationLog = async ({
  userId,
  voiceSampleId,
  score,
  threshold,
  accepted,
  decision,
}) => {
  const { rows } = await pool.query(
    `
      INSERT INTO verification_logs
        (user_id, voice_sample_id, score, threshold, accepted, decision)
      VALUES ($1, $2, $3, $4, $5, $6)
      RETURNING id
    `,
    [
      userId,
      voiceSampleId ?? null,
      score,
      threshold,
      Boolean(accepted),
      decision,
    ],
  )

  return getVerificationLogById(rows[0].id)
}

const getVerificationLogById = async (id) => {
  const { rows } = await pool.query(
    `
      SELECT
        id,
        user_id,
        voice_sample_id,
        score,
        threshold,
        accepted,
        decision,
        created_at
      FROM verification_logs
      WHERE id = $1
    `,
    [id],
  )

  if (!rows.length) {
    return null
  }

  return normalizeLog(rows[0])
}

const getVerificationLogsByUserId = async (userId, limit = 50) => {
  const { rows } = await pool.query(
    `
      SELECT
        vl.id,
        vl.user_id,
        vl.voice_sample_id,
        vl.score,
        vl.threshold,
        vl.accepted,
        vl.decision,
        vl.created_at,
        vs.file_path
      FROM verification_logs vl
      LEFT JOIN voice_samples vs ON vs.id = vl.voice_sample_id
      WHERE vl.user_id = $1
      ORDER BY vl.created_at DESC
      LIMIT $2
    `,
    [userId, Number(limit)],
  )

  return rows.map(normalizeLog)
}

const normalizeLog = (row) => ({
  id: row.id,
  user_id: row.user_id,
  voice_sample_id: row.voice_sample_id,
  file_path: row.file_path ?? null,
  score: Number(row.score),
  threshold: Number(row.threshold),
  accepted: Boolean(row.accepted),
  decision: row.decision,
  created_at: row.created_at,
})

module.exports = {
  createVerificationLog,
  getVerificationLogById,
  getVerificationLogsByUserId,
}
