const cosineSimilarity = (a, b) => {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length === 0) {
    throw new Error('Embeddings must be non-empty arrays of equal length')
  }

  let dot = 0
  let normA = 0
  let normB = 0

  for (let i = 0; i < a.length; i += 1) {
    const x = Number(a[i])
    const y = Number(b[i])
    dot += x * y
    normA += x * x
    normB += y * y
  }

  if (normA === 0 || normB === 0) {
    return -1
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

/**
 * Rank all templates by cosine similarity to the probe.
 * Returns best match plus second-best score for open-set / ambiguity gates.
 */
const findBestTemplateMatch = (probeEmbedding, templates) => {
  if (!Array.isArray(templates) || templates.length === 0) {
    return null
  }

  let best = null
  let secondScore = null

  for (const template of templates) {
    const score = cosineSimilarity(probeEmbedding, template.embedding)
    if (!best || score > best.score) {
      if (best) {
        secondScore = best.score
      }
      best = {
        user_id: template.user_id,
        username: template.username,
        score,
      }
    } else if (secondScore === null || score > secondScore) {
      secondScore = score
    }
  }

  return {
    ...best,
    second_score: secondScore,
    margin: secondScore === null ? null : best.score - secondScore,
  }
}

const decideByThreshold = (score, threshold) => {
  const accepted = score >= threshold
  return {
    score,
    threshold,
    accepted,
    decision: accepted ? 'ACCEPT' : 'REJECT',
  }
}

/**
 * Open-set identify gate: require a strong top match and (when available) a clear margin.
 */
const passesIdentifyGate = (
  match,
  { threshold, margin = 0 } = {},
) => {
  if (!match || typeof match.score !== 'number') {
    return {
      accepted: false,
      reason: 'no_match',
      score: null,
      second_score: null,
      margin: null,
      threshold,
      required_margin: margin,
    }
  }

  if (match.score < threshold) {
    return {
      accepted: false,
      reason: 'below_threshold',
      score: match.score,
      second_score: match.second_score,
      margin: match.margin,
      threshold,
      required_margin: margin,
    }
  }

  if (
    margin > 0 &&
    match.second_score !== null &&
    match.second_score !== undefined &&
    match.margin < margin
  ) {
    return {
      accepted: false,
      reason: 'ambiguous',
      score: match.score,
      second_score: match.second_score,
      margin: match.margin,
      threshold,
      required_margin: margin,
    }
  }

  return {
    accepted: true,
    reason: 'ok',
    score: match.score,
    second_score: match.second_score,
    margin: match.margin,
    threshold,
    required_margin: margin,
  }
}

module.exports = {
  cosineSimilarity,
  findBestTemplateMatch,
  decideByThreshold,
  passesIdentifyGate,
}
