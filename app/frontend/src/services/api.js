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
    if (
      error?.response?.status === 401 &&
      typeof unauthorizedHandler === 'function' &&
      !error?.config?.url?.includes('/auth/login')
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
