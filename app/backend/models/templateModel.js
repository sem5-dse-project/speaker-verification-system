const pgvector = require('pgvector/pg')
const { pool } = require('../config/db')

const MODEL_NAME = 'ecapa-tdnn'
const MODEL_VERSION = 'v1'

// Speaker embeddings are stored as pgvector `vector(192)` values (NOT json/text),
// so PostgreSQL can index and search them natively. Each user has at most one
// row with kind = 'template' (the averaged enrollment embedding used for
// verification/identification); the schema still allows multiple rows per
// user (e.g. future per-sample embeddings) via the `kind` column.
const upsertTemplate = async (userId, embedding, embeddingDim, numSamples, threshold = null) => {
  await pool.query(
    `
      INSERT INTO speaker_embeddings
        (user_id, embedding, embedding_dim, kind, model_name, model_version, num_samples, threshold, updated_at)
      VALUES ($1, $2, $3, 'template', $4, $5, $6, $7, now())
      ON CONFLICT (user_id) WHERE kind = 'template'
      DO UPDATE SET
        embedding = EXCLUDED.embedding,
        embedding_dim = EXCLUDED.embedding_dim,
        num_samples = EXCLUDED.num_samples,
        threshold = EXCLUDED.threshold,
        updated_at = now()
    `,
    [
      userId,
      pgvector.toSql(embedding),
      embeddingDim,
      MODEL_NAME,
      MODEL_VERSION,
      numSamples,
      threshold,
    ],
  )

  return getTemplateByUserId(userId)
}

const getTemplateByUserId = async (userId) => {
  const { rows } = await pool.query(
    `
      SELECT user_id, embedding, embedding_dim, num_samples, threshold, updated_at
      FROM speaker_embeddings
      WHERE user_id = $1 AND kind = 'template'
    `,
    [userId],
  )

  if (!rows.length) {
    return null
  }

  const row = rows[0]
  return {
    user_id: row.user_id,
    embedding: row.embedding,
    embedding_dim: row.embedding_dim,
    num_samples: row.num_samples,
    threshold: row.threshold,
    updated_at: row.updated_at,
  }
}

const getAllTemplatesWithUsers = async () => {
  const { rows } = await pool.query(
    `
      SELECT se.user_id, se.embedding, se.embedding_dim, se.threshold, u.username
      FROM speaker_embeddings se
      INNER JOIN users u ON u.id = se.user_id
      WHERE se.kind = 'template'
    `,
  )

  return rows.map((row) => ({
    user_id: row.user_id,
    username: row.username,
    embedding: row.embedding,
    embedding_dim: row.embedding_dim,
    threshold: row.threshold,
  }))
}

/**
 * Open-set identify: rank every enrolled template against the probe embedding
 * using pgvector's cosine-distance operator (`<=>`) and the HNSW index,
 * instead of pulling every embedding into Node and comparing in a loop.
 *
 * pgvector's `<=>` returns *cosine distance* (0 = identical, 2 = opposite),
 * which is `1 - cosine_similarity`. The rest of the app (threshold, decision
 * logic) is written in terms of cosine *similarity*, so we convert here with
 * `1 - distance` and keep the existing threshold semantics unchanged.
 */
const findBestMatch = async (probeEmbedding, limit = 2) => {
  const { rows } = await pool.query(
    `
      SELECT
        se.user_id,
        u.username,
        1 - (se.embedding <=> $1) AS score
      FROM speaker_embeddings se
      INNER JOIN users u ON u.id = se.user_id
      WHERE se.kind = 'template'
      ORDER BY se.embedding <=> $1
      LIMIT $2
    `,
    [pgvector.toSql(probeEmbedding), limit],
  )

  if (!rows.length) {
    return null
  }

  const best = rows[0]
  const second = rows[1] || null

  return {
    user_id: best.user_id,
    username: best.username,
    score: Number(best.score),
    second_score: second ? Number(second.score) : null,
    margin: second ? Number(best.score) - Number(second.score) : null,
  }
}

const deleteTemplateByUserId = async (userId) => {
  const result = await pool.query(
    "DELETE FROM speaker_embeddings WHERE user_id = $1 AND kind = 'template'",
    [userId],
  )
  return result.rowCount > 0
}

module.exports = {
  upsertTemplate,
  getTemplateByUserId,
  getAllTemplatesWithUsers,
  findBestMatch,
  deleteTemplateByUserId,
}

