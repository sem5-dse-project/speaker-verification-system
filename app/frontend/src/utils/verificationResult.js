/**
 * Format verification API result for UI status text.
 */
export function formatVerificationResult(result, fallbackMessage = 'Verification complete.') {
  const decision =
    result?.decision ||
    (result?.accepted === true ? 'ACCEPT' : result?.accepted === false ? 'REJECT' : null)

  const scoreText =
    typeof result?.score === 'number' ? ` (score ${result.score.toFixed(3)})` : ''

  if (!decision) {
    return {
      type: 'success',
      text: fallbackMessage,
    }
  }

  if (decision === 'REPLAY') {
    return {
      type: 'error',
      text: `Replay attack detected${scoreText}`,
    }
  }

  if (decision === 'UNCERTAIN') {
    return {
      type: 'warning',
      text: `Could not confidently check for replay — please re-record${scoreText}`,
    }
  }

  if (decision === 'NO_SPEECH') {
    return {
      type: 'warning',
      text: `No speech detected — please speak clearly and try again${scoreText}`,
    }
  }

  return {
    type: decision === 'REJECT' ? 'error' : 'success',
    text: `Verification ${decision}${scoreText}`,
  }
}
