import { describe, it, expect } from 'vitest'
import { formatVerificationResult } from './verificationResult.js'

describe('formatVerificationResult', () => {
  it('formats ACCEPT with score', () => {
    expect(
      formatVerificationResult({ decision: 'ACCEPT', score: 0.8123, accepted: true }),
    ).toEqual({
      type: 'success',
      text: 'Verification ACCEPT (score 0.812)',
    })
  })

  it('formats REJECT as error', () => {
    expect(
      formatVerificationResult({ decision: 'REJECT', score: 0.1, accepted: false }),
    ).toEqual({
      type: 'error',
      text: 'Verification REJECT (score 0.100)',
    })
  })

  it('infers decision from accepted boolean', () => {
    expect(formatVerificationResult({ accepted: true, score: 0.5 })).toEqual({
      type: 'success',
      text: 'Verification ACCEPT (score 0.500)',
    })
  })

  it('falls back when result is empty', () => {
    expect(formatVerificationResult(null, 'Done')).toEqual({
      type: 'success',
      text: 'Done',
    })
  })
})
