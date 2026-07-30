const jwt = require('jsonwebtoken')
const authMiddleware = require('../middleware/authMiddleware')
const { mockRes, mockNext } = require('./helpers')

describe('authMiddleware', () => {
  const secret = 'test-secret'

  beforeEach(() => {
    process.env.JWT_SECRET = secret
  })

  it('returns 401 when Authorization header is missing', () => {
    const req = { headers: {} }
    const res = mockRes()
    const next = mockNext()

    authMiddleware(req, res, next)

    expect(res.status).toHaveBeenCalledWith(401)
    expect(res.json).toHaveBeenCalledWith(
      expect.objectContaining({
        success: false,
        message: 'Authorization token missing',
      }),
    )
    expect(next).not.toHaveBeenCalled()
  })

  it('returns 401 when token is invalid', () => {
    const req = { headers: { authorization: 'Bearer not-a-real-token' } }
    const res = mockRes()
    const next = mockNext()

    authMiddleware(req, res, next)

    expect(res.status).toHaveBeenCalledWith(401)
    expect(res.json).toHaveBeenCalledWith(
      expect.objectContaining({
        success: false,
        message: 'Invalid or expired token',
      }),
    )
    expect(next).not.toHaveBeenCalled()
  })

  it('sets req.user and calls next for a valid token', () => {
    const token = jwt.sign({ id: 7, username: 'alice' }, secret, {
      expiresIn: '1h',
    })
    const req = { headers: { authorization: `Bearer ${token}` } }
    const res = mockRes()
    const next = mockNext()

    authMiddleware(req, res, next)

    expect(req.user).toEqual({ id: 7, username: 'alice' })
    expect(next).toHaveBeenCalled()
    expect(res.status).not.toHaveBeenCalled()
  })
})
