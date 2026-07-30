const fs = require('fs')
const os = require('os')
const path = require('path')
const { isPcmWavFile, buildEnrollmentTemplate } = require('../services/mlClient')

const makeTempFile = (name, contents) => {
  const filePath = path.join(os.tmpdir(), name)
  fs.writeFileSync(filePath, contents)
  return filePath
}

describe('mlClient', () => {
  let wavPath
  let junkPath

  beforeAll(() => {
    const header = Buffer.alloc(12)
    header.write('RIFF', 0)
    header.write('WAVE', 8)
    wavPath = makeTempFile(`mlclient-good-${Date.now()}.wav`, header)
    junkPath = makeTempFile(`mlclient-bad-${Date.now()}.webm`, Buffer.from('not-a-wav'))
  })

  afterAll(() => {
    for (const p of [wavPath, junkPath]) {
      try {
        fs.unlinkSync(p)
      } catch {
        /* ignore */
      }
    }
  })

  describe('isPcmWavFile', () => {
    it('returns true for RIFF/WAVE header', () => {
      expect(isPcmWavFile(wavPath)).toBe(true)
    })

    it('returns false for non-WAV bytes', () => {
      expect(isPcmWavFile(junkPath)).toBe(false)
    })

    it('returns false for missing file', () => {
      expect(isPcmWavFile(path.join(os.tmpdir(), 'missing-file.wav'))).toBe(false)
    })
  })

  describe('buildEnrollmentTemplate', () => {
    it('throws when no files are provided', async () => {
      await expect(buildEnrollmentTemplate([])).rejects.toThrow(
        'At least one enrollment audio file is required',
      )
    })

    it('throws when a file is not valid WAV', async () => {
      await expect(buildEnrollmentTemplate([junkPath])).rejects.toThrow(
        /not a valid WAV/,
      )
    })
  })
})
