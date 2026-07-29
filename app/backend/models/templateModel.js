const { pool } = require('../config/db')

const upsertTemplate = async (userId, embedding, embeddingDim, numSamples, threshold = null) => {
  const embeddingJson = JSON.stringify(embedding)

  await pool.query(
    `
      INSERT INTO enrollment_templates
        (user_id, embedding, embedding_dim, num_samples, threshold)
      VALUES (?, CAST(? AS JSON), ?, ?, ?)
      ON DUPLICATE KEY UPDATE
        embedding = VALUES(embedding),
        embedding_dim = VALUES(embedding_dim),
        num_samples = VALUES(num_samples),
        threshold = VALUES(threshold),
        updated_at = CURRENT_TIMESTAMP
    `,
    [userId, embeddingJson, embeddingDim, numSamples, threshold],
  )

  return getTemplateByUserId(userId)
}

const getTemplateByUserId = async (userId) => {
  const [rows] = await pool.query(
    `
      SELECT user_id, embedding, embedding_dim, num_samples, threshold, updated_at
      FROM enrollment_templates
      WHERE user_id = ?
    `,
    [userId],
  )

  if (!rows.length) {
    return null
  }

  const row = rows[0]
  let embedding = row.embedding
  if (typeof embedding === 'string') {
    embedding = JSON.parse(embedding)
  }

  return {
    user_id: row.user_id,
    embedding,
    embedding_dim: row.embedding_dim,
    num_samples: row.num_samples,
    threshold: row.threshold,
    updated_at: row.updated_at,
  }
}

const deleteTemplateByUserId = async (userId) => {
  const [result] = await pool.query(
    'DELETE FROM enrollment_templates WHERE user_id = ?',
    [userId],
  )
  return result.affectedRows > 0
}

module.exports = {
  upsertTemplate,
  getTemplateByUserId,
  deleteTemplateByUserId,
}
