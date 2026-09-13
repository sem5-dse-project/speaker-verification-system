jest.mock('../models/userModel')
jest.mock('bcrypt')
jest.mock('jsonwebtoken')

const bcrypt = require('bcrypt')
const jwt = require('jsonwebtoken')
const userModel = require('../models/userModel')
const { register, login } = require('../controllers/authController')
const { mockRes } = require('./helpers')

describe('authController', () => {
  beforeEach(() => {
    process.env.JWT_SECRET = 'test-secret'
  })

  describe('register', () => {
    it('returns 400 when username or password is missing', async () => {
      const req = { body: { username: 'bob' } }
      const res = mockRes()

      await register(req, res)

      expect(res.status).toHaveBeenCalledWith(400)
      expect(userModel.createUser).not.toHaveBeenCalled()
    })

    it('returns 409 when username already exists', async () => {
      userModel.findByUsername.mockResolvedValue({ id: 1, username: 'bob' })
      const req = { body: { username: 'bob', password: 'secret123' } }
      const res = mockRes()

      await register(req, res)

      expect(res.status).toHaveBeenCalledWith(409)
      expect(userModel.createUser).not.toHaveBeenCalled()
    })

    it('hashes password and creates user', async () => {
      userModel.findByUsername.mockResolvedValue(null)
      bcrypt.hash.mockResolvedValue('hashed')
      userModel.createUser.mockResolvedValue({ id: 2, username: 'bob' })

      const req = { body: { username: 'bob', password: 'secret123' } }
      const res = mockRes()

      await register(req, res)

      expect(bcrypt.hash).toHaveBeenCalledWith('secret123', 10)
      expect(userModel.createUser).toHaveBeenCalledWith('bob', 'hashed')
      expect(res.status).toHaveBeenCalledWith(201)
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({ success: true }),
      )
    })
  })

  describe('login', () => {
    it('returns 401 for unknown user', async () => {
      userModel.findByUsername.mockResolvedValue(null)
      const req = { body: { username: 'missing', password: 'x' } }
      const res = mockRes()

      await login(req, res)

      expect(res.status).toHaveBeenCalledWith(401)
    })

    it('returns token for valid credentials', async () => {
      userModel.findByUsername.mockResolvedValue({
        id: 3,
        username: 'bob',
        password: 'hashed',
        role: 'user',
      })
      bcrypt.compare.mockResolvedValue(true)
      jwt.sign.mockReturnValue('jwt-token')

      const req = { body: { username: 'bob', password: 'secret123' } }
      const res = mockRes()

      await login(req, res)

      expect(jwt.sign).toHaveBeenCalled()
      expect(res.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: true,
          token: 'jwt-token',
          user: { id: 3, username: 'bob', role: 'user' },
        }),
      )
    })
  })
})
