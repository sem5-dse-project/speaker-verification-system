const { Pool } = require('pg')
const pgvector = require('pgvector/pg')

const pool = new Pool({
  host: process.env.DATABASE_HOST,
  port: Number(process.env.DATABASE_PORT || 5432),
  user: process.env.DATABASE_USER,
  password: process.env.DATABASE_PASSWORD,
  database: process.env.DATABASE_NAME,
  max: 10,
  idleTimeoutMillis: 30000,
})

// Registers the `vector` type parser/serializer for every new connection so
// `pool.query` can bind/return plain JS number arrays for `vector(192)` columns.
// On a brand-new database the `vector` type doesn't exist until initSchema()
// runs `CREATE EXTENSION vector` below, so tolerate that and retry afterwards.
pool.on('connect', async (client) => {
  try {
    await pgvector.registerTypes(client)
  } catch (error) {
    console.warn('pgvector types not registered yet (extension missing?):', error.message)
  }
})

pool.on('error', (error) => {
  // Idle clients can be dropped by the server; log instead of crashing the app.
  console.error('Unexpected PostgreSQL client error:', error.message)
})

const EMBEDDING_DIM = 192

const initSchema = async () => {
  const client = await pool.connect()
  try {
    await client.query('CREATE EXTENSION IF NOT EXISTS vector')
    // Re-register on this client now that the type exists — it may be the
    // same connection that failed registration in the `connect` handler above.
    await pgvector.registerTypes(client)
  } finally {
    client.release()
  }

  await pool.query(`
    CREATE TABLE IF NOT EXISTS users (
      id BIGSERIAL PRIMARY KEY,
      username VARCHAR(50) UNIQUE NOT NULL,
      password VARCHAR(255) NOT NULL,
      role VARCHAR(16) NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `)

  await pool.query(`
    CREATE TABLE IF NOT EXISTS voice_samples (
      id BIGSERIAL PRIMARY KEY,
      user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      file_path VARCHAR(255) NOT NULL,
      sample_type VARCHAR(16) NOT NULL CHECK (sample_type IN ('enrollment', 'verification')),
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `)
  await pool.query('CREATE INDEX IF NOT EXISTS idx_voice_samples_user ON voice_samples(user_id)')

  // One user can have many speaker embeddings (per-model / per-version / future
  // per-sample rows). Exactly one row per user is the active verification
  // "template" (kind = 'template'), enforced by the partial unique index below.
  await pool.query(`
    CREATE TABLE IF NOT EXISTS speaker_embeddings (
      id BIGSERIAL PRIMARY KEY,
      user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      embedding vector(${EMBEDDING_DIM}) NOT NULL,
      embedding_dim INT NOT NULL DEFAULT ${EMBEDDING_DIM},
      kind VARCHAR(16) NOT NULL DEFAULT 'template' CHECK (kind IN ('template', 'sample')),
      model_name VARCHAR(64) NOT NULL DEFAULT 'ecapa-tdnn',
      model_version VARCHAR(32) NOT NULL DEFAULT 'v1',
      num_samples INT NULL,
      threshold DOUBLE PRECISION NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `)
  await pool.query(`
    CREATE UNIQUE INDEX IF NOT EXISTS ux_speaker_embeddings_template_per_user
      ON speaker_embeddings(user_id) WHERE kind = 'template'
  `)
  await pool.query(
    'CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_user ON speaker_embeddings(user_id)',
  )
  // HNSW + cosine ops: used by the open-set "identify" query, which ranks a
  // probe embedding against every enrolled template. HNSW is chosen over
  // IVFFlat because it needs no `lists` tuning tied to table size, gives
  // strong recall/latency for the modest number of enrolled users expected
  // here, and (unlike IVFFlat) does not degrade as new users enroll between
  // index rebuilds.
  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_hnsw_cosine
      ON speaker_embeddings USING hnsw (embedding vector_cosine_ops)
  `)

  await pool.query(`
    CREATE TABLE IF NOT EXISTS verification_logs (
      id BIGSERIAL PRIMARY KEY,
      user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      voice_sample_id BIGINT NULL REFERENCES voice_samples(id) ON DELETE SET NULL,
      score DOUBLE PRECISION NOT NULL,
      threshold DOUBLE PRECISION NOT NULL,
      accepted BOOLEAN NOT NULL,
      decision VARCHAR(16) NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `)
  await pool.query(
    'CREATE INDEX IF NOT EXISTS idx_verification_logs_user ON verification_logs(user_id)',
  )

  await pool.query(`
    CREATE TABLE IF NOT EXISTS collection_samples (
      id BIGSERIAL PRIMARY KEY,
      admin_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      speaker_id VARCHAR(64) NOT NULL,
      label VARCHAR(16) NOT NULL CHECK (label IN ('live', 'replay')),
      file_path VARCHAR(512) NOT NULL,
      phrase VARCHAR(255) NULL,
      phone_model VARCHAR(128) NULL,
      distance VARCHAR(64) NULL,
      volume VARCHAR(64) NULL,
      notes TEXT NULL,
      consent BOOLEAN NOT NULL DEFAULT true,
      replay_score DOUBLE PRECISION NULL,
      replay_decision VARCHAR(32) NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `)
}

module.exports = {
  pool,
  initSchema,
  EMBEDDING_DIM,
}

