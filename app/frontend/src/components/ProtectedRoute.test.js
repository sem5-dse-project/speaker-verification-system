import { describe, it, expect, beforeEach } from 'vitest'
import { hasValidToken } from './ProtectedRoute.jsx'

const makeToken = (payload) => {
  const header = btoa(JSON.stringify({ alg: 'none', typ: 'JWT' }))
  const body = btoa(JSON.stringify(payload))
  return `${header}.${body}.sig`
}

describe('hasValidToken', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('returns false when token is missing', () => {
    expect(hasValidToken()).toBe(false)
  })

  it('returns true for unexpired token', () => {
    const exp = Math.floor(Date.now() / 1000) + 3600
    localStorage.setItem('token', makeToken({ id: 1, exp }))
    expect(hasValidToken()).toBe(true)
  })

  it('returns false and clears storage for expired token', () => {
    const exp = Math.floor(Date.now() / 1000) - 10
    localStorage.setItem('token', makeToken({ id: 1, exp }))
    localStorage.setItem('user', '{"id":1}')

    expect(hasValidToken()).toBe(false)
    expect(localStorage.getItem('token')).toBeNull()
    expect(localStorage.getItem('user')).toBeNull()
  })

  it('returns false for malformed token', () => {
    localStorage.setItem('token', 'not-a-jwt')
    expect(hasValidToken()).toBe(false)
  })
})
