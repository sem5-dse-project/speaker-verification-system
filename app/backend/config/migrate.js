/**
 * Minimal, dependency-free SQL migration runner.
 *
 * Applies every *.sql file in migrations/ (sorted by filename) exactly once,
 * tracked in a `schema_migrations` table. No ORM: plain parameterized SQL.
 *
 * Usage: npm run migrate
 */
const fs = require('fs')
const path = require('path')
const dotenv = require('dotenv')

dotenv.config()

const { pool } = require('./db')

const MIGRATIONS_DIR = path.join(__dirname, '..', 'migrations')

const ensureMigrationsTable = async () => {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      name VARCHAR(255) PRIMARY KEY,
      applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `)
}

const getAppliedMigrations = async () => {
  const { rows } = await pool.query('SELECT name FROM schema_migrations')
  return new Set(rows.map((row) => row.name))
}

const runMigrations = async () => {
  await ensureMigrationsTable()
  const applied = await getAppliedMigrations()

  const files = fs
    .readdirSync(MIGRATIONS_DIR)
    .filter((name) => name.endsWith('.sql'))
    .sort()

  for (const file of files) {
    if (applied.has(file)) {
      console.log(`Skipping already-applied migration: ${file}`)
      continue
    }

    const sql = fs.readFileSync(path.join(MIGRATIONS_DIR, file), 'utf8')
    const client = await pool.connect()
    try {
      await client.query('BEGIN')
      await client.query(sql)
      await client.query('INSERT INTO schema_migrations (name) VALUES ($1)', [file])
      await client.query('COMMIT')
      console.log(`Applied migration: ${file}`)
    } catch (error) {
      await client.query('ROLLBACK')
      throw error
    } finally {
      client.release()
    }
  }
}

if (require.main === module) {
  runMigrations()
    .then(() => {
      console.log('Migrations complete')
      return pool.end()
    })
    .catch((error) => {
      console.error('Migration failed:', error.message)
      process.exitCode = 1
      return pool.end()
    })
}

module.exports = { runMigrations }
