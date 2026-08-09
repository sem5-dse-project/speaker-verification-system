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

  it('formats REPLAY as error', () => {
    expect(
      formatVerificationResult({ decision: 'REPLAY', score: 0.91, accepted: false }),
    ).toEqual({
      type: 'error',
      text: 'Replay attack detected (score 0.910)',
    })
  })

  it('formats UNCERTAIN as warning', () => {
    expect(
      formatVerificationResult({ decision: 'UNCERTAIN', score: 0.72, accepted: false }),
    ).toEqual({
      type: 'warning',
      text: 'Could not confidently check for spoof — please re-record (score 0.720)',
    })
  })

  it('formats SYNTHETIC as error with stage scores', () => {
    expect(
      formatVerificationResult({
        decision: 'SYNTHETIC',
        score: 0.95,
        accepted: false,
        replay: { score: 0.12 },
        la: { score: 0.95 },
      }),
    ).toEqual({
      type: 'error',
      text: 'Synthetic spoof detected (score 0.950) [replay 0.120 · LA 0.950]',
    })
  })

  it('formats NO_SPEECH as warning', () => {
    expect(
      formatVerificationResult({ decision: 'NO_SPEECH', score: 0, accepted: false }),
    ).toEqual({
      type: 'warning',
      text: 'No speech detected — please speak clearly and try again (score 0.000)',
    })
  })

  it('falls back when result is empty', () => {
    expect(formatVerificationResult(null, 'Done')).toEqual({
      type: 'success',
      text: 'Done',
    })
  })
})
