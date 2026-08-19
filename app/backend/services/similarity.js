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

const findBestTemplateMatch = (probeEmbedding, templates) => {
  if (!Array.isArray(templates) || templates.length === 0) {
    return null
  }

  let best = null

  for (const template of templates) {
    const score = cosineSimilarity(probeEmbedding, template.embedding)
    if (!best || score > best.score) {
      best = {
        user_id: template.user_id,
        username: template.username,
        score,
      }
    }
  }

  return best
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

module.exports = {
  cosineSimilarity,
  findBestTemplateMatch,
  decideByThreshold,
}