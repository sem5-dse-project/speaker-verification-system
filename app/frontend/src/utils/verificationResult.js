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

  return {
    type: decision === 'REJECT' || decision === 'REPLAY' ? 'error' : 'success',
    text: `Verification ${decision}${scoreText}`,
  }
}
