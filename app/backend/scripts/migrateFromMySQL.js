/**
 * One-off data migration: copies existing MariaDB/MySQL data into the new
 * PostgreSQL + pgvector schema. Safe to re-run (uses ON CONFLICT DO NOTHING
 * on primary keys), but intended to be run once against a fresh Postgres DB
 * that already has the schema applied (`npm run migrate`).
 *
 * Source (MariaDB/MySQL) connection is read from MYSQL_* env vars so it does
 * not collide with the app's own DATABASE_* (PostgreSQL) vars:
 *   MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
 *
 * Destination is the regular DATABASE_* PostgreSQL connection.
 *
 * Usage: npm run migrate:from-mysql
 */
const dotenv = require('dotenv')
const mysql = require('mysql2/promise')
const pgvector = require('pgvector/pg')

dotenv.config()

const { pool } = require('../config/db')

const openMysql = async () =>
  mysql.createConnection({
    host: process.env.MYSQL_HOST || 'localhost',
    port: Number(process.env.MYSQL_PORT || 3306),
    user: process.env.MYSQL_USER || 'root',
    password: process.env.MYSQL_PASSWORD || '',
    database: process.env.MYSQL_DATABASE || 'voice_authentication',
  })

// Resets a PostgreSQL identity/serial sequence to MAX(id) after inserting
// explicit ids, so future inserts continue from the right value.
const resyncSequence = async (table) => {
  await pool.query(
    `SELECT setval(pg_get_serial_sequence($1, 'id'), COALESCE((SELECT MAX(id) FROM ${table}), 1))`,
    [table],
  )
}

const migrateUsers = async (mysqlConn) => {
  const [rows] = await mysqlConn.query('SELECT id, username, password, role, created_at FROM users')
  for (const row of rows) {
    await pool.query(
      `INSERT INTO users (id, username, password, role, created_at)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (id) DO NOTHING`,
      [row.id, row.username, row.password, row.role || 'user', row.created_at],
    )
  }
  await resyncSequence('users')
  console.log(`Migrated ${rows.length} users`)
}

const migrateVoiceSamples = async (mysqlConn) => {
  const [rows] = await mysqlConn.query(
    'SELECT id, user_id, file_path, sample_type, created_at FROM voice_samples',
  )
  for (const row of rows) {
    await pool.query(
      `INSERT INTO voice_samples (id, user_id, file_path, sample_type, created_at)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (id) DO NOTHING`,
      [row.id, row.user_id, row.file_path, row.sample_type, row.created_at],
    )
  }
  await resyncSequence('voice_samples')
  console.log(`Migrated ${rows.length} voice_samples`)
}

// enrollment_templates.embedding was stored as JSON text; convert to a
// pgvector `vector(192)` value (never as JSON/text/array column).
const migrateEnrollmentTemplates = async (mysqlConn) => {
  const [rows] = await mysqlConn.query(
    'SELECT user_id, embedding, embedding_dim, num_samples, threshold, updated_at FROM enrollment_templates',
  )
  for (const row of rows) {
    const embedding = typeof row.embedding === 'string' ? JSON.parse(row.embedding) : row.embedding
    await pool.query(
      `INSERT INTO speaker_embeddings
         (user_id, embedding, embedding_dim, kind, model_name, model_version, num_samples, threshold, updated_at)
       VALUES ($1, $2, $3, 'template', 'ecapa-tdnn', 'v1', $4, $5, $6)
       ON CONFLICT (user_id) WHERE kind = 'template' DO NOTHING`,
      [row.user_id, pgvector.toSql(embedding), row.embedding_dim, row.num_samples, row.threshold, row.updated_at],
    )
  }
  console.log(`Migrated ${rows.length} enrollment templates`)
}

const migrateVerificationLogs = async (mysqlConn) => {
  const [rows] = await mysqlConn.query(
    `SELECT id, user_id, voice_sample_id, score, threshold, accepted, decision, created_at
     FROM verification_logs`,
  )
  for (const row of rows) {
    await pool.query(
      `INSERT INTO verification_logs
         (id, user_id, voice_sample_id, score, threshold, accepted, decision, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
       ON CONFLICT (id) DO NOTHING`,
      [
        row.id,
        row.user_id,
        row.voice_sample_id,
        row.score,
        row.threshold,
        Boolean(row.accepted),
        row.decision,
        row.created_at,
      ],
    )
  }
  await resyncSequence('verification_logs')
  console.log(`Migrated ${rows.length} verification_logs`)
}

const migrateCollectionSamples = async (mysqlConn) => {
  const [rows] = await mysqlConn.query('SELECT * FROM collection_samples')
  for (const row of rows) {
    await pool.query(
      `INSERT INTO collection_samples
         (id, admin_id, speaker_id, label, file_path, phrase, phone_model, distance, volume,
          notes, consent, replay_score, replay_decision, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
       ON CONFLICT (id) DO NOTHING`,
      [
        row.id,
        row.admin_id,
        row.speaker_id,
        row.label,
        row.file_path,
        row.phrase,
        row.phone_model,
        row.distance,
        row.volume,
        row.notes,
        Boolean(row.consent),
        row.replay_score,
        row.replay_decision,
        row.created_at,
      ],
    )
  }
  await resyncSequence('collection_samples')
  console.log(`Migrated ${rows.length} collection_samples`)
}

const run = async () => {
  const mysqlConn = await openMysql()
  try {
    // Order matters: users first (foreign keys), then dependents.
    await migrateUsers(mysqlConn)
    await migrateVoiceSamples(mysqlConn)
    await migrateEnrollmentTemplates(mysqlConn)
    await migrateVerificationLogs(mysqlConn)
    await migrateCollectionSamples(mysqlConn)
  } finally {
    await mysqlConn.end()
    await pool.end()
  }
}

if (require.main === module) {
  run()
    .then(() => console.log('MySQL -> PostgreSQL data migration complete'))
    .catch((error) => {
      console.error('Data migration failed:', error.message)
      process.exitCode = 1
    })
}

module.exports = { run }
