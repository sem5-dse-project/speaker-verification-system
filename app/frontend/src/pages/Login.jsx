import { useMemo, useState } from 'react'
import { AtSign, LockKeyhole, Eye, EyeOff, ShieldCheck } from 'lucide-react'
import AuthCard from '../components/AuthCard.jsx'
import InputField from '../components/InputField.jsx'

const INITIAL_FORM = {
  username: '',
  password: '',
  rememberMe: false,
}

function Login() {
  const [form, setForm] = useState(INITIAL_FORM)
  const [showPassword, setShowPassword] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const errors = useMemo(() => {
    const nextErrors = {
      username: '',
      password: '',
    }

    if (submitted && form.username.trim().length < 3) {
      nextErrors.username = 'Username must be at least 3 characters.'
    }

    if (submitted && form.password.length < 8) {
      nextErrors.password = 'Password must be at least 8 characters.'
    }

    return nextErrors
  }, [form.username, form.password, submitted])

  const hasErrors = Object.values(errors).some(Boolean)
  const canSubmit = !isSubmitting

  const handleFieldChange = (field) => (event) => {
    const value = event.target.type === 'checkbox' ? event.target.checked : event.target.value
    setForm((previous) => ({ ...previous, [field]: value }))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setSubmitted(true)

    if (form.username.trim().length < 3 || form.password.length < 8) {
      return
    }

    setIsSubmitting(true)
    await new Promise((resolve) => setTimeout(resolve, 800))
    setIsSubmitting(false)
  }

  return (
    <main className="h-screen overflow-hidden bg-gradient-to-b from-slate-50 via-slate-100 to-blue-50/40 px-4 py-4 sm:px-6 sm:py-6 lg:px-8 lg:py-8">
      <div className="mx-auto flex h-full max-w-6xl items-center justify-center">
        <AuthCard
          title="Welcome Back"
          subtitle="Sign in to access your voice authentication dashboard."
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
              helperText={errors.username || 'Enter your account username.'}
            />

            <InputField
              id="password"
              label="Password"
              type={showPassword ? 'text' : 'password'}
              placeholder="Enter your password"
              value={form.password}
              onChange={handleFieldChange('password')}
              icon={LockKeyhole}
              error={errors.password}
              helperText={errors.password || 'Use the password linked to your voice profile.'}
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

            <div className="flex items-center justify-between pt-1 text-sm">
              <label className="flex cursor-pointer items-center gap-2 text-slate-600">
                <input
                  type="checkbox"
                  checked={form.rememberMe}
                  onChange={handleFieldChange('rememberMe')}
                  className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-300"
                />
                Remember Me
              </label>

              <a href="#" className="font-medium text-blue-600 transition-colors hover:text-blue-500">
                Forgot Password?
              </a>
            </div>

            <button
              type="submit"
              disabled={!canSubmit}
              className="mt-1 inline-flex w-full items-center justify-center rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-200/60 transition-all duration-200 hover:bg-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 focus-visible:ring-offset-1 disabled:cursor-not-allowed disabled:bg-blue-400"
            >
              {isSubmitting ? 'Signing In...' : 'Sign In'}
            </button>

            <p className="pt-1 text-center text-sm text-slate-600">
              Don't have an account?{' '}
              <a
                href="#"
                className="font-semibold text-blue-600 transition-colors hover:text-blue-500"
              >
                Create Account
              </a>
            </p>

            <div className="rounded-xl border border-blue-100 bg-blue-50/70 px-3 py-2.5 text-sm text-blue-800">
              <p className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4" />
                Your credentials and voice data are securely protected.
              </p>
            </div>

            {submitted && !hasErrors && !isSubmitting && (
              <p className="rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
                Mock authentication successful. Dashboard access is ready.
              </p>
            )}
          </form>
        </AuthCard>
      </div>
    </main>
  )
}

export default Login