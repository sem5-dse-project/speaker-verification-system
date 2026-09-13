const { pool } = require('../config/db')

const createSample = async ({
  adminId,
  speakerId,
  label,
  filePath,
  phrase,
  phoneModel,
  distance,
  volume,
  notes,
  consent,
  replayScore,
  replayDecision,
}) => {
  const [result] = await pool.query(
    `INSERT INTO collection_samples (
      admin_id, speaker_id, label, file_path, phrase, phone_model,
      distance, volume, notes, consent, replay_score, replay_decision
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      adminId,
      speakerId,
      label,
      filePath,
      phrase || null,
      phoneModel || null,
      distance || null,
      volume || null,
      notes || null,
      consent ? 1 : 0,
      replayScore ?? null,
      replayDecision || null,
    ],
  )

  return { id: result.insertId }
}

const listSamples = async () => {
  const [rows] = await pool.query(
    `SELECT
      cs.id,
      cs.speaker_id,
      cs.label,
      cs.file_path,
      cs.phrase,
      cs.phone_model,
      cs.distance,
      cs.volume,
      cs.notes,
      cs.consent,
      cs.replay_score,
      cs.replay_decision,
      cs.created_at,
      u.username AS collected_by
    FROM collection_samples cs
    JOIN users u ON u.id = cs.admin_id
    ORDER BY cs.created_at DESC`,
  )

  return rows
}

const countByLabel = async () => {
  const [rows] = await pool.query(
    `SELECT label, COUNT(*) AS count
     FROM collection_samples
     GROUP BY label`,
  )

  return rows
}

module.exports = {
  createSample,
  listSamples,
  countByLabel,
}
