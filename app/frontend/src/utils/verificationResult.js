/**
 * Format verification API result for UI status text.
 */
function stageScoreLine(result) {
  const parts = []
  const replayScore = result?.replay?.score
  const laScore = result?.la?.score
  if (typeof replayScore === 'number') {
    parts.push(`replay ${replayScore.toFixed(3)}`)
  }
  if (typeof laScore === 'number') {
    parts.push(`LA ${laScore.toFixed(3)}`)
  }
  return parts.length ? ` [${parts.join(' · ')}]` : ''
}

export function formatVerificationResult(result, fallbackMessage = 'Verification complete.') {
  const decision =
    result?.decision ||
    (result?.accepted === true ? 'ACCEPT' : result?.accepted === false ? 'REJECT' : null)

  const scoreText =
    typeof result?.score === 'number' ? ` (score ${result.score.toFixed(3)})` : ''
  const stages = stageScoreLine(result)

  if (!decision) {
    return {
      type: 'success',
      text: fallbackMessage,
    }
  }

  if (decision === 'REPLAY') {
    return {
      type: 'error',
      text: `Replay attack detected${scoreText}${stages}`,
    }
  }

  if (decision === 'SYNTHETIC') {
    return {
      type: 'error',
      text: `Synthetic spoof detected${scoreText}${stages}`,
    }
  }

  if (decision === 'UNCERTAIN') {
    return {
      type: 'warning',
      text: `Could not confidently check for spoof — please re-record${scoreText}${stages}`,
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
    text: `Verification ${decision}${scoreText}${stages}`,
  }
}
