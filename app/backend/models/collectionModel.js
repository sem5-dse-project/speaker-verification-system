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
  const { rows } = await pool.query(
    `INSERT INTO collection_samples (
      admin_id, speaker_id, label, file_path, phrase, phone_model,
      distance, volume, notes, consent, replay_score, replay_decision
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    RETURNING id`,
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
      Boolean(consent),
      replayScore ?? null,
      replayDecision || null,
    ],
  )

  return { id: rows[0].id }
}

const listSamples = async () => {
  const { rows } = await pool.query(
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
  const { rows } = await pool.query(
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
