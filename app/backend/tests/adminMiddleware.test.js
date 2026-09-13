const adminMiddleware = require('../middleware/adminMiddleware')
const { mockRes, mockNext } = require('./helpers')

describe('adminMiddleware', () => {
  it('returns 403 for non-admin users', () => {
    const req = { user: { id: 1, username: 'bob', role: 'user' } }
    const res = mockRes()
    const next = mockNext()

    adminMiddleware(req, res, next)

    expect(res.status).toHaveBeenCalledWith(403)
    expect(next).not.toHaveBeenCalled()
  })

  it('calls next for admin users', () => {
    const req = { user: { id: 2, username: 'admin1', role: 'admin' } }
    const res = mockRes()
    const next = mockNext()

    adminMiddleware(req, res, next)

    expect(next).toHaveBeenCalled()
    expect(res.status).not.toHaveBeenCalled()
  })
})
