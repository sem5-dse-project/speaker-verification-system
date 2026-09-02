const crypto = require('crypto')

const TTL_SECONDS = Number(process.env.VOICE_LOGIN_CACHE_TTL_SECONDS || 300)
const cache = new Map()

const nowMs = () => Date.now()

const purgeExpired = () => {
  const now = nowMs()
  for (const [token, entry] of cache.entries()) {
    if (entry.expires_at_ms <= now) {
      cache.delete(token)
    }
  }
}

const createToken = () => {
  if (typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return crypto.randomBytes(24).toString('hex')
}

const createVoiceLoginSession = ({
  probe_embedding,
  embedding_dim,
  identified_user_id,
  identified_username,
  identify_score,
  replay,
}) => {
  purgeExpired()

  const token = createToken()
  const createdAtMs = nowMs()
  const expiresAtMs = createdAtMs + TTL_SECONDS * 1000

  cache.set(token, {
    probe_embedding,
    embedding_dim,
    identified_user_id,
    identified_username,
    identify_score,
    replay,
    created_at_ms: createdAtMs,
    expires_at_ms: expiresAtMs,
  })

  return {
    token,
    expires_at: new Date(expiresAtMs).toISOString(),
    ttl_seconds: TTL_SECONDS,
  }
}

const getVoiceLoginSession = (token) => {
  purgeExpired()
  const entry = cache.get(token)
  if (!entry) {
    return null
  }

  return {
    ...entry,
    expires_at: new Date(entry.expires_at_ms).toISOString(),
  }
}

const deleteVoiceLoginSession = (token) => {
  cache.delete(token)
}

module.exports = {
  createVoiceLoginSession,
  getVoiceLoginSession,
  deleteVoiceLoginSession,
}