const { pool } = require('../config/db')

const createUser = async (username, passwordHash, role = 'user') => {
  const [result] = await pool.query(
    'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
    [username, passwordHash, role],
  )

  return {
    id: result.insertId,
    username,
    role,
  }
}

const findByUsername = async (username) => {
  const [rows] = await pool.query(
    'SELECT id, username, password, role FROM users WHERE username = ?',
    [username],
  )

  return rows[0] || null
}

const findById = async (id) => {
  const [rows] = await pool.query(
    'SELECT id, username, role, created_at FROM users WHERE id = ?',
    [id],
  )

  return rows[0] || null
}

const findAuthById = async (id) => {
  const [rows] = await pool.query(
    'SELECT id, username, password, role FROM users WHERE id = ?',
    [id],
  )

  return rows[0] || null
}

const listAdmins = async () => {
  const [rows] = await pool.query(
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
