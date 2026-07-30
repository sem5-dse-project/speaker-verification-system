const { sampleIndexFromName, toTimestampMs } = require('../middleware/upload')

describe('upload naming helpers', () => {
  describe('sampleIndexFromName', () => {
    it('parses enroll_N filenames', () => {
      expect(sampleIndexFromName('enroll_1_123.wav')).toBe('1')
      expect(sampleIndexFromName('enroll_3_999.wav')).toBe('3')
    })

    it('parses verify_N filenames', () => {
      expect(sampleIndexFromName('verify_1_abc.wav')).toBe('1')
    })

    it('returns null when index is missing', () => {
      expect(sampleIndexFromName('random.wav')).toBeNull()
      expect(sampleIndexFromName('')).toBeNull()
      expect(sampleIndexFromName(null)).toBeNull()
    })
  })

  describe('toTimestampMs', () => {
    it('returns YYYYMMDD_HHMMSSmmm format', () => {
      expect(toTimestampMs()).toMatch(/^\d{8}_\d{9}$/)
    })
  })
})
