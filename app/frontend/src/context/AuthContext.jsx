import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { setUnauthorizedHandler } from '../services/api.js'

const TOKEN_KEY = 'token'
const USER_KEY = 'user'

const readStorage = (storage) => {
  const token = storage.getItem(TOKEN_KEY)
  const userRaw = storage.getItem(USER_KEY)

  if (!token || !userRaw) {
    return null
  }

  try {
    return {
      token,
      user: JSON.parse(userRaw),
    }
  } catch {
    storage.removeItem(TOKEN_KEY)
    storage.removeItem(USER_KEY)
    return null
  }
}

const readStoredAuth = () => {
  const localAuth = readStorage(localStorage)
  if (localAuth) {
    return localAuth
  }
  return readStorage(sessionStorage)
}

const writeStoredAuth = ({ token, user, rememberMe }) => {
  const primary = rememberMe ? localStorage : sessionStorage
  const secondary = rememberMe ? sessionStorage : localStorage

  primary.setItem(TOKEN_KEY, token)
  primary.setItem(USER_KEY, JSON.stringify(user))

  secondary.removeItem(TOKEN_KEY)
  secondary.removeItem(USER_KEY)
}

const clearStoredAuth = () => {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
  sessionStorage.removeItem(TOKEN_KEY)
  sessionStorage.removeItem(USER_KEY)
}

const AuthContext = createContext(null)

function AuthProvider({ children }) {
  const [auth, setAuth] = useState(() => readStoredAuth())

  const login = useCallback((token, user, rememberMe = false) => {
    const nextAuth = { token, user }
    writeStoredAuth({ ...nextAuth, rememberMe })
    setAuth(nextAuth)
  }, [])

  const logout = useCallback(() => {
    clearStoredAuth()
    setAuth(null)
  }, [])

  useEffect(() => {
    setUnauthorizedHandler(() => {
      clearStoredAuth()
      setAuth(null)
      if (window.location.pathname !== '/login') {
        window.location.assign('/login')
      }
    })

    return () => {
      setUnauthorizedHandler(null)
    }
  }, [])

  const value = useMemo(
    () => ({
      token: auth?.token || null,
      user: auth?.user || null,
      isAuthenticated: Boolean(auth?.token),
      login,
      logout,
    }),
    [auth, login, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}

export { AuthProvider, useAuth }