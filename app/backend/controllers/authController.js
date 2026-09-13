const bcrypt = require('bcrypt')
const jwt = require('jsonwebtoken')
const userModel = require('../models/userModel')

const register = async (req, res) => {
  try {
    const { username, password } = req.body

    if (!username || !password) {
      return res.status(400).json({
        success: false,
        message: 'username and password are required',
      })
    }

    const existingUser = await userModel.findByUsername(username)
    if (existingUser) {
      return res.status(409).json({
        success: false,
        message: 'Username already exists',
      })
    }

    const passwordHash = await bcrypt.hash(password, 10)
    await userModel.createUser(username, passwordHash)

    return res.status(201).json({
      success: true,
      message: 'User registered successfully',
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to register user',
      error: error.message,
    })
  }
}

const login = async (req, res) => {
  try {
    const { username, password } = req.body

    if (!username || !password) {
      return res.status(400).json({
        success: false,
        message: 'username and password are required',
      })
    }

    const user = await userModel.findByUsername(username)
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials',
      })
    }

    const isMatch = await bcrypt.compare(password, user.password)
    if (!isMatch) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials',
      })
    }

    const token = jwt.sign(
      { id: user.id, username: user.username, role: user.role || 'user' },
      process.env.JWT_SECRET,
      { expiresIn: '1d' },
    )

    return res.json({
      success: true,
      token,
      user: {
        id: user.id,
        username: user.username,
        role: user.role || 'user',
      },
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to login',
      error: error.message,
    })
  }
}

module.exports = {
  register,
  login,
}
