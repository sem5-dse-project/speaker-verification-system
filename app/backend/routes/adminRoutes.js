const express = require('express')
const authMiddleware = require('../middleware/authMiddleware')
const adminMiddleware = require('../middleware/adminMiddleware')
const { collectionUpload } = require('../middleware/upload')
const adminController = require('../controllers/adminController')

const router = express.Router()

router.use(authMiddleware, adminMiddleware)

router.post('/admins', adminController.createAdmin)
router.get('/admins', adminController.listAdmins)

router.post(
  '/collection',
  collectionUpload.single('audio'),
  adminController.uploadCollectionSample,
)
router.get('/collection', adminController.listCollectionSamples)
router.get('/collection/export', adminController.exportCollectionMetadata)

module.exports = router
