import { useMemo, useState } from 'react'
import { AtSign, LockKeyhole, Eye, EyeOff, Check } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import AuthCard from '../components/AuthCard.jsx'
import InputField from '../components/InputField.jsx'
import PasswordStrength from '../components/PasswordStrength.jsx'
import api from '../services/api.js'

const INITIAL_FORM = {
  username: '',
  password: '',
}

function Register() {
  const navigate = useNavigate()
  const [form, setForm] = useState(INITIAL_FORM)
  const [showPassword, setShowPassword] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [statusMessage, setStatusMessage] = useState({ type: '', text: '' })

  const requirements = useMemo(
    () => [
      {
        id: 'length',
        label: 'At least 8 characters',
        valid: form.password.length >= 8,
      },
      {
        id: 'uppercase',
        label: 'One uppercase letter',
        valid: /[A-Z]/.test(form.password),
      },
      {
        id: 'lowercase',
        label: 'One lowercase letter',
        valid: /[a-z]/.test(form.password),
      },
      {
        id: 'number',
        label: 'One number',
        valid: /\d/.test(form.password),
      },
    ],
    [form.password],
  )

  const errors = useMemo(() => {
    const nextErrors = {
      username: '',
      password: '',
    }

    if (submitted && form.username.trim().length < 3) {
      nextErrors.username = 'Username must be at least 3 characters.'
    }

    if (submitted && !requirements.every((requirement) => requirement.valid)) {
      nextErrors.password = 'Password does not meet all requirements.'
    }

    return nextErrors
  }, [form, submitted, requirements])

  const handleFieldChange = (field) => (event) => {
    const { value } = event.target
    setForm((previous) => ({ ...previous, [field]: value }))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setSubmitted(true)
    setStatusMessage({ type: '', text: '' })

    if (form.username.trim().length < 3 || !requirements.every((requirement) => requirement.valid)) {
      return
    }

    setIsSubmitting(true)

    try {
      const response = await api.post('/auth/register', {
        username: form.username.trim(),
        password: form.password,
      })

      setStatusMessage({
        type: 'success',
        text: response.data?.message || 'User registered successfully',
      })

      setTimeout(() => {
        navigate('/login', { replace: true })
      }, 900)
    } catch (error) {
      setStatusMessage({
        type: 'error',
        text: error.response?.data?.message || 'Registration failed. Please try again.',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="h-screen overflow-hidden bg-gradient-to-b from-slate-50 via-slate-100 to-blue-50/40 px-4 py-4 sm:px-6 sm:py-6 lg:px-8 lg:py-8">
      <div className="mx-auto flex h-full max-w-6xl items-center justify-center">
        <AuthCard
          title="Create Account"
          subtitle="Create an account to enroll and verify your voice."
        >
          <form className="space-y-3" onSubmit={handleSubmit} noValidate>
            <InputField
              id="username"
              label="Username"
              placeholder="jane.doe"
              value={form.username}
              onChange={handleFieldChange('username')}
              icon={AtSign}
              error={errors.username}
              helperText={errors.username || 'Choose a unique username.'}
            />

            <InputField
              id="password"
              label="Password"
              type={showPassword ? 'text' : 'password'}
              placeholder="Create a strong password"
              value={form.password}
              onChange={handleFieldChange('password')}
              icon={LockKeyhole}
              error={errors.password}
              helperText={errors.password || 'Make it secure and memorable.'}
              rightElement={
                <button
                  type="button"
                  onClick={() => setShowPassword((previous) => !previous)}
                  className="rounded-lg p-1.5 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              }
            />

            <PasswordStrength password={form.password} />

            <ul className="grid gap-1 text-xs text-slate-600 sm:grid-cols-2">
              {requirements.map((requirement) => (
                <li key={requirement.id} className="flex items-center gap-2">
                  <span
                    className={[
                      'inline-flex h-4 w-4 items-center justify-center rounded-full border transition-colors',
                      requirement.valid
                        ? 'border-emerald-300 bg-emerald-100 text-emerald-700'
                        : 'border-slate-300 bg-slate-100 text-slate-400',
                    ].join(' ')}
                  >
                    <Check className="h-3 w-3" />
                  </span>
                  {requirement.label}
                </li>
              ))}
            </ul>

            <button
              type="submit"
              disabled={isSubmitting}
              className="mt-1 inline-flex w-full items-center justify-center rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-200/60 transition-all duration-200 hover:bg-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 focus-visible:ring-offset-1"
            >
              {isSubmitting ? 'Creating Account...' : 'Create Account'}
            </button>

            <p className="pt-1 text-center text-sm text-slate-600">
              Already have an account?{' '}
              <Link
                to="/login"
                className="font-semibold text-blue-600 transition-colors hover:text-blue-500"
              >
                Sign In
              </Link>
            </p>

            {statusMessage.type === 'success' && (
              <p className="rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
                {statusMessage.text}
              </p>
            )}

            {statusMessage.type === 'error' && (
              <p className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
                {statusMessage.text}
              </p>
            )}
          </form>
        </AuthCard>
      </div>
    </main>
  )
}

export default Register