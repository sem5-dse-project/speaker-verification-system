-- Initial PostgreSQL + pgvector schema for the speaker authentication system.
-- Mirrors config/db.js#initSchema (kept in sync manually); this file is the
-- explicit, reviewable migration artifact requested by the migration tooling.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
  id BIGSERIAL PRIMARY KEY,
  username VARCHAR(50) UNIQUE NOT NULL,
  password VARCHAR(255) NOT NULL,
  role VARCHAR(16) NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS voice_samples (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  file_path VARCHAR(255) NOT NULL,
  sample_type VARCHAR(16) NOT NULL CHECK (sample_type IN ('enrollment', 'verification')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_voice_samples_user ON voice_samples(user_id);

-- One row per user with kind = 'template' holds the active averaged
-- enrollment embedding used for verification/identification. The table can
-- also hold other kinds of rows (e.g. 'sample') so a user can be associated
-- with multiple speaker embeddings, per the schema requirement.
CREATE TABLE IF NOT EXISTS speaker_embeddings (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  embedding vector(192) NOT NULL,
  embedding_dim INT NOT NULL DEFAULT 192,
  kind VARCHAR(16) NOT NULL DEFAULT 'template' CHECK (kind IN ('template', 'sample')),
  model_name VARCHAR(64) NOT NULL DEFAULT 'ecapa-tdnn',
  model_version VARCHAR(32) NOT NULL DEFAULT 'v1',
  num_samples INT NULL,
  threshold DOUBLE PRECISION NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_speaker_embeddings_template_per_user
  ON speaker_embeddings(user_id) WHERE kind = 'template';

CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_user ON speaker_embeddings(user_id);

-- HNSW + cosine distance ops for the open-set identify query (probe vs. every
-- enrolled template). Chosen over IVFFlat because:
--   * HNSW needs no `lists` parameter tuned to table size and keeps good
--     recall as new users enroll, without periodic re-tuning/rebuilding.
--   * IVFFlat is trained on the data present at CREATE INDEX time and its
--     recall degrades for rows inserted afterwards until it is rebuilt -
--     not a good fit for a table that grows continuously as users enroll.
--   * The number of enrolled speakers in this system is small/medium, where
--     HNSW's higher build cost is not a concern and its query latency and
--     recall are both better than IVFFlat's.
CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_hnsw_cosine
  ON speaker_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS verification_logs (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  voice_sample_id BIGINT NULL REFERENCES voice_samples(id) ON DELETE SET NULL,
  score DOUBLE PRECISION NOT NULL,
  threshold DOUBLE PRECISION NOT NULL,
  accepted BOOLEAN NOT NULL,
  decision VARCHAR(16) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_verification_logs_user ON verification_logs(user_id);

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
);
