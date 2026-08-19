import { describe, it, expect } from 'vitest'
import { buildVerificationDetails, formatVerificationResult } from './verificationResult.js'

describe('formatVerificationResult', () => {
  it('formats ACCEPT with score', () => {
    expect(
      formatVerificationResult({ decision: 'ACCEPT', score: 0.8123, accepted: true }),
    ).toMatchObject({
      type: 'success',
      text: 'Verification ACCEPT (score 0.812)',
    })
  })

  it('formats REJECT as error', () => {
    expect(
      formatVerificationResult({ decision: 'REJECT', score: 0.1, accepted: false }),
    ).toMatchObject({
      type: 'error',
      text: 'Verification REJECT (score 0.100)',
    })
  })

  it('infers decision from accepted boolean', () => {
    expect(formatVerificationResult({ accepted: true, score: 0.5 })).toMatchObject({
      type: 'success',
      text: 'Verification ACCEPT (score 0.500)',
    })
  })

  it('formats REPLAY as error', () => {
    expect(
      formatVerificationResult({ decision: 'REPLAY', score: 0.91, accepted: false }),
    ).toMatchObject({
      type: 'error',
      text: 'Replay attack detected (score 0.910)',
    })
  })

  it('formats UNCERTAIN as warning', () => {
    expect(
      formatVerificationResult({ decision: 'UNCERTAIN', score: 0.72, accepted: false }),
    ).toMatchObject({
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
    ).toMatchObject({
      type: 'error',
      text: 'Synthetic spoof detected (score 0.950) [replay 0.120 · LA 0.950]',
    })
  })

  it('formats NO_SPEECH as warning', () => {
    expect(
      formatVerificationResult({ decision: 'NO_SPEECH', score: 0, accepted: false }),
    ).toMatchObject({
      type: 'warning',
      text: 'No speech detected — please speak clearly and try again (score 0.000)',
    })
  })

  it('falls back when result is empty', () => {
    expect(formatVerificationResult(null, 'Done')).toMatchObject({
      type: 'success',
      text: 'Done',
    })
  })

  it('builds stage details for live verification', () => {
    expect(
      buildVerificationDetails({
        decision: 'ACCEPT',
        score: 0.82,
        threshold: 0.25,
        accepted: true,
        replay: {
          score: 0.12,
          threshold: 0.15,
          threshold_low: 0.05,
          threshold_high: 0.25,
          decision: 'LIVE',
          feature_type: 'inverted_mel',
          is_replay: false,
          accepted: true,
        },
      }),
    ).toEqual([
      expect.objectContaining({
        label: 'Voice detected',
        status: 'success',
      }),
      expect.objectContaining({
        label: 'Replay attack',
        status: 'success',
      }),
      expect.objectContaining({
        label: 'Speaker verification',
        status: 'success',
      }),
    ])
  })

  it('marks speaker verification skipped after replay rejection', () => {
    expect(
      buildVerificationDetails({
        decision: 'REPLAY',
        score: 0.91,
        threshold: 0.74,
        accepted: false,
        replay: {
          score: 0.91,
          threshold: 0.74,
          threshold_low: 0.64,
          threshold_high: 0.84,
          decision: 'REPLAY',
          feature_type: 'inverted_mel',
          is_replay: true,
          accepted: false,
        },
      }),
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: 'Speaker verification',
          status: 'neutral',
          summary: expect.stringContaining('skipped'),
        }),
      ]),
    )
  })
})
