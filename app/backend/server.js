const express = require('express')
const cors = require('cors')
const dotenv = require('dotenv')
const path = require('path')

dotenv.config()

const { pool, initSchema } = require('./config/db')
const authRoutes = require('./routes/authRoutes')
const userRoutes = require('./routes/userRoutes')
const voiceRoutes = require('./routes/voiceRoutes')

const app = express()
const PORT = process.env.PORT || 5000

app.use(
  cors({
    origin: 'http://localhost:5173',
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  }),
)

app.use(express.json())
app.use('/uploads', express.static(path.join(__dirname, 'uploads')))

app.get('/api/health', (_req, res) => {
  res.json({
    success: true,
    message: 'Voice authentication backend is running',
  })
})

app.get('/api/health/db', async (_req, res) => {
  try {
    await pool.query('SELECT 1')

    return res.json({
      success: true,
      message: 'Database connection is healthy',
    })
  } catch (error) {
    return res.status(503).json({
      success: false,
      message: 'Database connection failed',
      error: error.message,
    })
  }
})

app.use('/api/auth', authRoutes)
app.use('/api/users', userRoutes)
app.use('/api/voice', voiceRoutes)

app.use((error, _req, res, _next) => {
  if (error && error.message === 'Only WAV audio files are allowed') {
    return res.status(400).json({
      success: false,
      message: error.message,
    })
  }

  return res.status(500).json({
    success: false,
    message: 'Unexpected server error',
    error: error.message,
  })
})

const startServer = async () => {
  try {
    await pool.query('SELECT 1')
    await initSchema()

    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`)
    })
  } catch (error) {
    console.error('Failed to start server:', error.message)
    process.exit(1)
  }
}

startServer()
