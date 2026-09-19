const { pool } = require('../config/db')

const createUser = async (username, passwordHash, role = 'user') => {
  const { rows } = await pool.query(
    'INSERT INTO users (username, password, role) VALUES ($1, $2, $3) RETURNING id, username, role',
    [username, passwordHash, role],
  )

  return rows[0]
}

const findByUsername = async (username) => {
  const { rows } = await pool.query(
    'SELECT id, username, password, role FROM users WHERE username = $1',
    [username],
  )

  return rows[0] || null
}

const findById = async (id) => {
  const { rows } = await pool.query(
    'SELECT id, username, role, created_at FROM users WHERE id = $1',
    [id],
  )

  return rows[0] || null
}

const findAuthById = async (id) => {
  const { rows } = await pool.query(
    'SELECT id, username, password, role FROM users WHERE id = $1',
    [id],
  )

  return rows[0] || null
}

const listAdmins = async () => {
  const { rows } = await pool.query(
    `SELECT id, username, created_at
     FROM users
     WHERE role = 'admin'
     ORDER BY created_at ASC`,
  )

  return rows
}

module.exports = {
  createUser,
  findByUsername,
  findById,
  findAuthById,
  listAdmins,
}
