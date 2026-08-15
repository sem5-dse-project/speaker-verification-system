const decodeBase64Url = (value) => {
  const normalized = value.replace(/-/g, '+').replace(/_/g, '/')
  const padding = normalized.length % 4
  const padded = padding ? normalized.padEnd(normalized.length + (4 - padding), '=') : normalized
  return atob(padded)
}

const decodeJwtPayload = (token) => {
  if (!token || typeof token !== 'string') {
    return null
  }

  const parts = token.split('.')
  if (parts.length < 2) {
    return null
  }

  try {
    return JSON.parse(decodeBase64Url(parts[1]))
  } catch {
    return null
  }
}

const isTokenValid = (token) => {
  const payload = decodeJwtPayload(token)
  if (!payload) {
    return false
  }

  if (!payload.exp) {
    return true
  }

  return payload.exp * 1000 > Date.now()
}

export { decodeJwtPayload, isTokenValid }