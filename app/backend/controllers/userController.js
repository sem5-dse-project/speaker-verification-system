const userModel = require('../models/userModel')

const getProfile = async (req, res) => {
  try {
    const user = await userModel.findById(req.user.id)

    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found',
      })
    }

    return res.json({
      success: true,
      user,
    })
  } catch (error) {
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch profile',
      error: error.message,
    })
  }
}

module.exports = {
  getProfile,
}
