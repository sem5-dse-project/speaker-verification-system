/**
 * Data and Database Integrity Testing
 *
 * Exercises the PostgreSQL + pgvector schema and its access methods directly
 * (models + raw SQL), independent of the HTTP/UI layer, so incorrect schema
 * behavior or data corruption is caught before it reaches the API.
 *
 * Requires a real database (see docker-compose.yml) reachable via the
 * DATABASE_* env vars / .env — run `docker compose up -d` first. Unlike the
 * unit tests under tests/, `config/db` is NOT mocked here.
 */
const dotenv = require('dotenv')

dotenv.config()

const { pool, initSchema } = require('../../config/db')
const userModel = require('../../models/userModel')
const voiceModel = require('../../models/voiceModel')
const templateModel = require('../../models/templateModel')
const verificationLogModel = require('../../models/verificationLogModel')
const collectionModel = require('../../models/collectionModel')

const EMBEDDING_DIM = 192

const makeEmbedding = (fillValue = 0.1) => new Array(EMBEDDING_DIM).fill(fillValue)

const createTestUser = async (usernameSuffix, role = 'user') =>
  userModel.createUser(`db_integrity_${usernameSuffix}`, 'hashed-password', role)

beforeAll(async () => {
  await initSchema()
})

afterEach(async () => {
  // Reset all tables between tests so each test seeds its own valid/invalid
  // data without interference (independent DB-level state per the technique).
  await pool.query(
    'TRUNCATE users, voice_samples, speaker_embeddings, verification_logs, collection_samples RESTART IDENTITY CASCADE',
  )
})

afterAll(async () => {
  await pool.end()
})

describe('Database Integrity: users', () => {
  it('persists a valid user and reads it back unchanged', async () => {
    const created = await userModel.createUser('db_integrity_alice', 'hashed-password', 'user')
    const fetched = await userModel.findById(created.id)

    expect(fetched.username).toBe('db_integrity_alice')
    expect(fetched.role).toBe('user')
  })

  it('rejects a duplicate username (unique constraint)', async () => {
    await createTestUser('dup')
    await expect(createTestUser('dup')).rejects.toThrow(/duplicate key/i)
  })

  it('rejects an invalid role (CHECK constraint)', async () => {
    await expect(
      pool.query("INSERT INTO users (username, password, role) VALUES ($1, $2, $3)", [
        'db_integrity_badrole',
        'hashed-password',
        'superadmin',
      ]),
    ).rejects.toThrow(/violates check constraint/i)
  })
})

describe('Database Integrity: voice_samples', () => {
  it('links a valid sample to its user and lists it in history', async () => {
    const user = await createTestUser('voice_owner')
    await voiceModel.createVoiceSample(user.id, '/uploads/sample1.wav', 'enrollment')

    const history = await voiceModel.getVoiceHistoryByUserId(user.id)
    expect(history).toHaveLength(1)
    expect(history[0].sample_type).toBe('enrollment')
  })

  it('rejects an invalid sample_type (CHECK constraint)', async () => {
    const user = await createTestUser('voice_badtype')
    await expect(
      voiceModel.createVoiceSample(user.id, '/uploads/sample1.wav', 'not_a_real_type'),
    ).rejects.toThrow(/violates check constraint/i)
  })

  it('cascades deletes from users to voice_samples', async () => {
    const user = await createTestUser('voice_cascade')
    await voiceModel.createVoiceSample(user.id, '/uploads/sample1.wav', 'enrollment')

    await pool.query('DELETE FROM users WHERE id = $1', [user.id])

    const { rows } = await pool.query('SELECT * FROM voice_samples WHERE user_id = $1', [user.id])
    expect(rows).toHaveLength(0)
  })
})

describe('Database Integrity: speaker_embeddings (pgvector)', () => {
  it('round-trips a 192-dim embedding through upsert/read without precision loss', async () => {
    const user = await createTestUser('template_roundtrip')
    const embedding = new Array(EMBEDDING_DIM).fill(0).map((_, i) => (i % 2 === 0 ? 0.25 : -0.25))

    await templateModel.upsertTemplate(user.id, embedding, EMBEDDING_DIM, 3, 0.45)
    const stored = await templateModel.getTemplateByUserId(user.id)

    expect(stored.embedding).toHaveLength(EMBEDDING_DIM)
    stored.embedding.forEach((value, i) => {
      expect(value).toBeCloseTo(embedding[i], 5)
    })
  })

  it('rejects an embedding with the wrong dimension', async () => {
    const user = await createTestUser('template_baddim')
    await expect(
      templateModel.upsertTemplate(user.id, makeEmbedding().slice(0, 64), 64, 1, 0.45),
    ).rejects.toThrow(/expected 192 dimensions, not 64/i)
  })

  it('rejects an invalid kind (CHECK constraint)', async () => {
    const user = await createTestUser('template_badkind')
    const pgvector = require('pgvector/pg')
    await expect(
      pool.query(
        `INSERT INTO speaker_embeddings (user_id, embedding, embedding_dim, kind)
         VALUES ($1, $2, $3, 'not_a_real_kind')`,
        [user.id, pgvector.toSql(makeEmbedding()), EMBEDDING_DIM],
      ),
    ).rejects.toThrow(/violates check constraint/i)
  })

  it('enforces at most one template row per user (partial unique index)', async () => {
    const user = await createTestUser('template_unique')
    const pgvector = require('pgvector/pg')
    await templateModel.upsertTemplate(user.id, makeEmbedding(), EMBEDDING_DIM, 3, 0.45)

    await expect(
      pool.query(
        `INSERT INTO speaker_embeddings (user_id, embedding, embedding_dim, kind)
         VALUES ($1, $2, $3, 'template')`,
        [user.id, pgvector.toSql(makeEmbedding(0.2)), EMBEDDING_DIM],
      ),
    ).rejects.toThrow(/duplicate key/i)
  })

  it('cascades deletes from users to speaker_embeddings', async () => {
    const user = await createTestUser('template_cascade')
    await templateModel.upsertTemplate(user.id, makeEmbedding(), EMBEDDING_DIM, 3, 0.45)

    await pool.query('DELETE FROM users WHERE id = $1', [user.id])

    const stored = await templateModel.getTemplateByUserId(user.id)
    expect(stored).toBeNull()
  })

  it('ranks templates by true cosine similarity via the HNSW <=> query', async () => {
    const identical = await createTestUser('template_identical')
    const orthogonal = await createTestUser('template_orthogonal')

    const probe = [1, ...new Array(EMBEDDING_DIM - 1).fill(0)]
    const identicalEmbedding = [1, ...new Array(EMBEDDING_DIM - 1).fill(0)]
    const orthogonalEmbedding = [0, 1, ...new Array(EMBEDDING_DIM - 2).fill(0)]

    await templateModel.upsertTemplate(identical.id, identicalEmbedding, EMBEDDING_DIM, 1, 0.45)
    await templateModel.upsertTemplate(orthogonal.id, orthogonalEmbedding, EMBEDDING_DIM, 1, 0.45)

    const match = await templateModel.findBestMatch(probe, 2)

    expect(match.user_id).toBe(identical.id)
    expect(match.score).toBeCloseTo(1, 4)
    expect(match.second_score).toBeCloseTo(0, 4)
  })
})

describe('Database Integrity: verification_logs', () => {
  it('persists a valid log and joins the originating voice sample', async () => {
    const user = await createTestUser('log_owner')
    const sample = await voiceModel.createVoiceSample(user.id, '/uploads/verify1.wav', 'verification')

    await verificationLogModel.createVerificationLog({
      userId: user.id,
      voiceSampleId: sample.id,
      score: 0.82,
      threshold: 0.45,
      accepted: true,
      decision: 'ACCEPT',
    })

    const [log] = await verificationLogModel.getVerificationLogsByUserId(user.id)
    expect(log.file_path).toBe('/uploads/verify1.wav')
    expect(log.accepted).toBe(true)
  })

  it('rejects a log missing the required score (NOT NULL constraint)', async () => {
    const user = await createTestUser('log_badscore')
    await expect(
      verificationLogModel.createVerificationLog({
        userId: user.id,
        voiceSampleId: null,
        score: null,
        threshold: 0.45,
        accepted: false,
        decision: 'REJECT',
      }),
    ).rejects.toThrow(/violates not-null constraint/i)
  })

  it('nulls voice_sample_id (not the log) when the referenced sample is deleted', async () => {
    const user = await createTestUser('log_setnull')
    const sample = await voiceModel.createVoiceSample(user.id, '/uploads/verify2.wav', 'verification')
    const created = await verificationLogModel.createVerificationLog({
      userId: user.id,
      voiceSampleId: sample.id,
      score: 0.5,
      threshold: 0.45,
      accepted: true,
      decision: 'ACCEPT',
    })

    await pool.query('DELETE FROM voice_samples WHERE id = $1', [sample.id])

    const log = await verificationLogModel.getVerificationLogById(created.id)
    expect(log).not.toBeNull()
    expect(log.voice_sample_id).toBeNull()
  })

  it('cascades deletes from users to verification_logs', async () => {
    const user = await createTestUser('log_cascade')
    const created = await verificationLogModel.createVerificationLog({
      userId: user.id,
      voiceSampleId: null,
      score: 0.5,
      threshold: 0.45,
      accepted: true,
      decision: 'ACCEPT',
    })

    await pool.query('DELETE FROM users WHERE id = $1', [user.id])

    const log = await verificationLogModel.getVerificationLogById(created.id)
    expect(log).toBeNull()
  })
})

describe('Database Integrity: collection_samples', () => {
  it('persists a valid live/replay sample', async () => {
    const admin = await createTestUser('collect_admin', 'admin')
    const { id } = await collectionModel.createSample({
      adminId: admin.id,
      speakerId: 'spk001',
      label: 'live',
      filePath: '/uploads/collect1.wav',
      consent: true,
    })

    // BIGSERIAL ids come back from pg as strings to avoid precision loss.
    expect(Number(id)).toBeGreaterThan(0)
  })

  it('rejects an invalid label (CHECK constraint)', async () => {
    const admin = await createTestUser('collect_badlabel', 'admin')
    await expect(
      collectionModel.createSample({
        adminId: admin.id,
        speakerId: 'spk002',
        label: 'not_a_real_label',
        filePath: '/uploads/collect2.wav',
        consent: true,
      }),
    ).rejects.toThrow(/violates check constraint/i)
  })
})
