import { describe, it, expect, beforeEach, vi } from 'vitest'
import api from '../services/api.js'

describe('api interceptor', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('attaches Bearer token when present', async () => {
    localStorage.setItem('token', 'abc123')

    const handlers = api.interceptors.request.handlers
    const fulfilled = handlers.find((h) => h?.fulfilled)?.fulfilled
    expect(fulfilled).toBeTypeOf('function')

    const config = await fulfilled({ headers: {} })
    expect(config.headers.Authorization).toBe('Bearer abc123')
  })

  it('leaves Authorization unset when token is missing', async () => {
    const handlers = api.interceptors.request.handlers
    const fulfilled = handlers.find((h) => h?.fulfilled)?.fulfilled

    const config = await fulfilled({ headers: {} })
    expect(config.headers.Authorization).toBeUndefined()
  })
})
