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

function toMetricValue(value, digits = 3) {
  if (typeof value === 'number') {
    return value.toFixed(digits)
  }

  if (typeof value === 'boolean') {
    return value ? 'yes' : 'no'
  }

  if (value === null || value === undefined) {
    return null
  }

  return String(value)
}

function buildMetric(label, value, digits) {
  const formatted = toMetricValue(value, digits)
  return formatted === null ? null : { label, value: formatted }
}

function buildStage(label, status, summary, metrics = []) {
  return {
    label,
    status,
    summary,
    metrics: metrics.filter(Boolean),
  }
}

function decisionStatus(decision) {
  if (decision === 'ACCEPT' || decision === 'LIVE') {
    return 'success'
  }

  if (decision === 'REJECT' || decision === 'REPLAY' || decision === 'SYNTHETIC') {
    return 'error'
  }

  if (decision === 'UNCERTAIN' || decision === 'NO_SPEECH') {
    return 'warning'
  }

  return 'neutral'
}

function describeReplayStage(replay) {
  const decision = replay?.decision || (replay?.is_replay ? 'REPLAY' : 'LIVE')

  if (decision === 'NO_SPEECH') {
    return buildStage(
      'Voice detected',
      'warning',
      'No speech detected, so the verification pipeline stopped early.',
      [
        buildMetric('RMS', replay?.rms, 4),
        buildMetric('Decision', decision, 0),
      ],
    )
  }

  return buildStage(
    'Voice detected',
    'success',
    'Speech was detected and the sample moved to replay screening.',
    [
      buildMetric('RMS', replay?.rms, 4),
      buildMetric('Decision', decision, 0),
    ],
  )
}

function describeReplayAttackStage(replay) {
  if (!replay) {
    return null
  }

  const decision = replay.decision || (replay.is_replay ? 'REPLAY' : 'LIVE')
  const status = decisionStatus(decision)
  const summaryMap = {
    REPLAY: 'Replay characteristics were detected and the sample was blocked.',
    LIVE: 'Replay signals stayed below the detection threshold.',
    UNCERTAIN: 'Replay screening was inconclusive and the sample should be re-recorded.',
    NO_SPEECH: 'Replay screening stopped because no speech was detected.',
  }

  return buildStage(
    'Replay attack',
    status,
    summaryMap[decision] || 'Replay screening completed for this sample.',
    [
      buildMetric('Score', replay.score),
      buildMetric('Threshold', replay.threshold),
      buildMetric('Low threshold', replay.threshold_low),
      buildMetric('High threshold', replay.threshold_high),
      buildMetric('Feature type', replay.feature_type, 0),
      buildMetric('Replay flagged', replay.is_replay),
      buildMetric('Accepted', replay.accepted),
    ],
  )
}

function describeSpoofStage(la) {
  if (!la) {
    return null
  }

  const decision = la.decision || (la.is_synthetic ? 'SYNTHETIC' : 'LIVE')
  const status = decisionStatus(decision)
  const summaryMap = {
    SYNTHETIC: 'Synthetic speech signals were detected.',
    LIVE: 'Synthetic speech checks stayed clear.',
    UNCERTAIN: 'Synthetic speech checks were inconclusive.',
  }

  return buildStage(
    'Synthetic / LA',
    status,
    summaryMap[decision] || 'Synthetic speech checks completed for this sample.',
    [
      buildMetric('Score', la.score),
      buildMetric('Threshold', la.threshold),
      buildMetric('Decision', decision, 0),
      buildMetric('Accepted', la.accepted),
    ],
  )
}

function describeSpeakerStage(result) {
  const decision = result?.decision || (result?.accepted === true ? 'ACCEPT' : 'REJECT')
  const skipped = ['NO_SPEECH', 'REPLAY', 'SYNTHETIC', 'UNCERTAIN'].includes(decision)

  if (skipped) {
    const reasonMap = {
      NO_SPEECH: 'Speaker verification was skipped because no speech was detected.',
      REPLAY: 'Speaker verification was skipped after replay detection blocked the sample.',
      SYNTHETIC: 'Speaker verification was skipped after synthetic speech was detected.',
      UNCERTAIN: 'Speaker verification was skipped because the sample needs to be re-recorded.',
    }

    return buildStage(
      'Speaker verification',
      'neutral',
      reasonMap[decision] || 'Speaker verification was skipped for this sample.',
      [
        buildMetric('Score', result?.score),
        buildMetric('Threshold', result?.threshold),
        buildMetric('Decision', decision, 0),
      ],
    )
  }

  return buildStage(
    'Speaker verification',
    decisionStatus(decision),
    result?.accepted
      ? 'The enrolled speaker template accepted the sample.'
      : 'The enrolled speaker template rejected the sample.',
    [
      buildMetric('Score', result?.score),
      buildMetric('Threshold', result?.threshold),
      buildMetric('Accepted', result?.accepted),
      buildMetric('Decision', decision, 0),
    ],
  )
}

export function buildVerificationDetails(result) {
  if (!result) {
    return []
  }

  const stages = [
    describeReplayStage(result.replay),
    describeReplayAttackStage(result.replay),
    describeSpoofStage(result.la),
    describeSpeakerStage(result),
  ]

  return stages.filter(Boolean)
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
      details: buildVerificationDetails(result),
      decision: null,
      score: result?.score,
    }
  }

  if (decision === 'REPLAY') {
    return {
      type: 'error',
      text: `Replay attack detected${scoreText}${stages}`,
      details: buildVerificationDetails(result),
      decision,
      score: result?.score,
    }
  }

  if (decision === 'SYNTHETIC') {
    return {
      type: 'error',
      text: `Synthetic spoof detected${scoreText}${stages}`,
      details: buildVerificationDetails(result),
      decision,
      score: result?.score,
    }
  }

  if (decision === 'UNCERTAIN') {
    return {
      type: 'warning',
      text: `Could not confidently check for spoof — please re-record${scoreText}${stages}`,
      details: buildVerificationDetails(result),
      decision,
      score: result?.score,
    }
  }

  if (decision === 'NO_SPEECH') {
    return {
      type: 'warning',
      text: `No speech detected — please speak clearly and try again${scoreText}`,
      details: buildVerificationDetails(result),
      decision,
      score: result?.score,
    }
  }

  return {
    type: decision === 'REJECT' ? 'error' : 'success',
    text: `Verification ${decision}${scoreText}${stages}`,
    details: buildVerificationDetails(result),
    decision,
    score: result?.score,
  }
}
