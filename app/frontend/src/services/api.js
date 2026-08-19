import axios from 'axios'

let unauthorizedHandler = null

const fromEnv = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api'

const api = axios.create({
  baseURL: fromEnv.replace(/\/$/, ''),
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token') || sessionStorage.getItem('token')

  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }

  return config
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const requestUrl = error?.config?.url || ''
    const hasAuthHeader = Boolean(error?.config?.headers?.Authorization)
    const isPublicAuthFlow =
      requestUrl.includes('/auth/login') ||
      requestUrl.includes('/voice/login') ||
      requestUrl.includes('/voice/identify')

    if (
      error?.response?.status === 401 &&
      typeof unauthorizedHandler === 'function' &&
      hasAuthHeader &&
      !isPublicAuthFlow
    ) {
      unauthorizedHandler(error)
    }

    return Promise.reject(error)
  },
)

const setUnauthorizedHandler = (handler) => {
  unauthorizedHandler = handler
}

export { setUnauthorizedHandler }
export default api
