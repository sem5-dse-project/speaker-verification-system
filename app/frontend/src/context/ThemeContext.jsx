import { createContext, useContext, useEffect, useMemo, useState } from 'react'

const THEME_KEY = 'theme'

const getPreferredTheme = () => {
  const stored = localStorage.getItem(THEME_KEY)
  if (stored === 'light' || stored === 'dark') {
    return stored
  }
  if (typeof window.matchMedia === 'function') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }
  return 'light'
}

const ThemeContext = createContext(null)

function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => getPreferredTheme())

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme((previous) => (previous === 'dark' ? 'light' : 'dark'))
  }

  const value = useMemo(() => ({ theme, toggleTheme }), [theme])

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

const useTheme = () => {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider')
  }
  return context
}

export { ThemeProvider, useTheme }
