import { useMemo, useState } from 'react'
import { AtSign, LockKeyhole, ShieldCheck } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import AuthCard from '../components/AuthCard.jsx'
import InputField from '../components/InputField.jsx'
import PageShell from '../components/PageShell.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import api from '../services/api.js'

const INITIAL_FORM = {
  username: '',
  password: '',
  rememberMe: false,
}

function AdminLogin() {
  const navigate = useNavigate()
  const { login } = useAuth()
  const [form, setForm] = useState(INITIAL_FORM)
  const [submitted, setSubmitted] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [statusMessage, setStatusMessage] = useState({ type: '', text: '' })

  const errors = useMemo(() => {
    const nextErrors = {
      username: '',
      password: '',
    }

    if (submitted && form.username.trim().length < 3) {
      nextErrors.username = 'Username must be at least 3 characters.'
    }

    if (submitted && form.password.length < 1) {
      nextErrors.password = 'Password is required.'
    }

    return nextErrors
  }, [form.username, form.password, submitted])

  const handleFieldChange = (field) => (event) => {
    const value = event.target.type === 'checkbox' ? event.target.checked : event.target.value
    setForm((previous) => ({ ...previous, [field]: value }))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setSubmitted(true)
    setStatusMessage({ type: '', text: '' })

    if (form.username.trim().length < 3 || form.password.length < 1) {
      return
    }

    setIsSubmitting(true)

    try {
      const response = await api.post('/auth/login', {
        username: form.username.trim(),
        password: form.password,
      })

      const user = response.data.user
      if (user?.role !== 'admin') {
        setStatusMessage({
          type: 'error',
          text: 'This account is not an admin. Use the normal login page.',
        })
        return
      }

      login(response.data.token, user, form.rememberMe)
      navigate('/admin', { replace: true })
    } catch (error) {
      setStatusMessage({
        type: 'error',
        text: error.response?.data?.message || 'Admin login failed.',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <PageShell variant="admin">
      <div className="mx-auto flex max-w-xl items-center justify-center">
        <AuthCard
          title="Admin Login"
          subtitle="Sign in to collect live vs replay research audio."
        >
          <form className="space-y-3" onSubmit={handleSubmit} noValidate>
            <InputField
              id="admin-username"
              label="Username"
              placeholder="admin1"
              value={form.username}
              onChange={handleFieldChange('username')}
              icon={AtSign}
              error={errors.username}
            />

            <InputField
              id="admin-password"
              label="Password"
              type="password"
              placeholder="Enter admin password"
              value={form.password}
              onChange={handleFieldChange('password')}
              icon={LockKeyhole}
              error={errors.password}
            />

            <label className="flex cursor-pointer items-center gap-2 text-sm text-muted">
              <input
                type="checkbox"
                checked={form.rememberMe}
                onChange={handleFieldChange('rememberMe')}
                className="checkbox-brand"
              />
              Remember Me
            </label>

            <button type="submit" disabled={isSubmitting} className="btn-primary-lg">
              {isSubmitting ? 'Signing In...' : 'Sign In as Admin'}
            </button>

            <Link to="/login" className="btn-secondary-lg">
              Back to User Login
            </Link>

            <div className="alert-info">
              <p className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4" />
                Default seed account: admin1 / admin1234 (change after first login).
              </p>
            </div>

            {statusMessage.type === 'error' && (
              <p className="alert-error">{statusMessage.text}</p>
            )}
          </form>
        </AuthCard>
      </div>
    </PageShell>
  )
}

export default AdminLogin
