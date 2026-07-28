const { pool } = require('../config/db')

const createUser = async (username, passwordHash) => {
  const [result] = await pool.query(
    'INSERT INTO users (username, password) VALUES (?, ?)',
    [username, passwordHash],
  )

  return {
    id: result.insertId,
    username,
  }
}

const findByUsername = async (username) => {
  const [rows] = await pool.query(
    'SELECT id, username, password FROM users WHERE username = ?',
    [username],
  )

  return rows[0] || null
}

const findById = async (id) => {
  const [rows] = await pool.query(
    'SELECT id, username, created_at FROM users WHERE id = ?',
    [id],
  )

  return rows[0] || null
}

module.exports = {
  createUser,
  findByUsername,
  findById,
}
