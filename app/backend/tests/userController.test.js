jest.mock('../models/userModel')

const userModel = require('../models/userModel')
const { getProfile } = require('../controllers/userController')
const { mockRes } = require('./helpers')

describe('userController.getProfile', () => {
  it('returns 404 when user is missing', async () => {
    userModel.findById.mockResolvedValue(null)
    const req = { user: { id: 99 } }
    const res = mockRes()

    await getProfile(req, res)

    expect(res.status).toHaveBeenCalledWith(404)
  })

  it('returns user profile', async () => {
    const user = { id: 1, username: 'alice', created_at: '2026-01-01' }
    userModel.findById.mockResolvedValue(user)
    const req = { user: { id: 1 } }
    const res = mockRes()

    await getProfile(req, res)

    expect(res.json).toHaveBeenCalledWith({ success: true, user })
  })
})
