const bcrypt = require('bcrypt')
const { pool } = require('./db')

const DEFAULT_ADMIN_USERNAME = 'admin1'
const DEFAULT_ADMIN_PASSWORD = 'admin1234'

const seedDefaultAdmin = async () => {
  const passwordHash = await bcrypt.hash(DEFAULT_ADMIN_PASSWORD, 10)
  const [rows] = await pool.query(
    'SELECT id FROM users WHERE username = ?',
    [DEFAULT_ADMIN_USERNAME],
  )

  if (rows.length === 0) {
    await pool.query(
      'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
      [DEFAULT_ADMIN_USERNAME, passwordHash, 'admin'],
    )
    console.log(`Seeded default admin user "${DEFAULT_ADMIN_USERNAME}"`)
    return
  }

  await pool.query(
    'UPDATE users SET password = ?, role = ? WHERE username = ?',
    [passwordHash, 'admin', DEFAULT_ADMIN_USERNAME],
  )
  console.log(`Updated default admin "${DEFAULT_ADMIN_USERNAME}" password and role`)
}

module.exports = {
  seedDefaultAdmin,
  DEFAULT_ADMIN_USERNAME,
  DEFAULT_ADMIN_PASSWORD,
}
