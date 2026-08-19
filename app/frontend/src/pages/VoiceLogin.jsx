import { useState } from 'react'
import { ArrowLeft, AlertCircle, CheckCircle2, Mic, UserRound } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import Recorder from '../components/Recorder.jsx'
import PrimaryButton from '../components/PrimaryButton.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import api from '../services/api.js'

function VoiceLogin() {
  const navigate = useNavigate()
  const { login } = useAuth()

  const [recording, setRecording] = useState(null)
  const [identifyResult, setIdentifyResult] = useState(null)
  const [password, setPassword] = useState('')
  const [rememberMe, setRememberMe] = useState(false)
  const [isIdentifying, setIsIdentifying] = useState(false)
  const [isSigningIn, setIsSigningIn] = useState(false)
  const [statusMessage, setStatusMessage] = useState({ type: '', text: '' })

  const resetFlow = () => {
    setIdentifyResult(null)
    setPassword('')
    setStatusMessage({ type: '', text: '' })
  }

  const handleIdentify = async () => {
    if (!recording?.blob) {
      return
    }

    setIsIdentifying(true)
    setStatusMessage({ type: '', text: '' })

    try {
      const formData = new FormData()
      const wavLikeFile = new File([recording.blob], `voice_login_${Date.now()}.wav`, {
        type: 'audio/wav',
      })
      formData.append('audio', wavLikeFile)

      const response = await api.post('/voice/identify', formData)
      setIdentifyResult(response.data)
      setStatusMessage({
        type: 'success',
        text: 'Voice identified. Confirm with your password to continue.',
      })
    } catch (error) {
      const message = error.response?.data?.message || 'Voice identification failed.'
      setStatusMessage({ type: 'error', text: message })
      setIdentifyResult(null)
    } finally {
      setIsIdentifying(false)
    }
  }

  const handleVoiceLogin = async (event) => {
    event.preventDefault()
    if (!identifyResult?.temporary_login_token || password.length < 8) {
      return
    }

    setIsSigningIn(true)
    setStatusMessage({ type: '', text: '' })

    try {
      const response = await api.post('/voice/login', {
        temporary_login_token: identifyResult.temporary_login_token,
        password,
      })

      login(response.data.token, response.data.user, rememberMe)
      navigate('/dashboard', { replace: true })
    } catch (error) {
      setStatusMessage({
        type: 'error',
        text: error.response?.data?.message || 'Voice login failed. Please try again.',
      })
    } finally {
      setIsSigningIn(false)
    }
  }

  const userGuess = identifyResult?.identified_user
  const guessScore = identifyResult?.similarity_score

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-100 to-white px-4 py-10 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-4xl space-y-6">
        <div>
          <button
            type="button"
            onClick={() => navigate('/login')}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Login
          </button>
        </div>

        <header className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">Login with Voice</h1>
          <p className="text-base text-slate-600 sm:text-lg">
            Record once, pass replay detection, then confirm your password.
          </p>
        </header>

        <Recorder
          onRecordingChange={(value) => {
            setRecording(value)
            resetFlow()
          }}
          onRecorderError={(message) =>
            setStatusMessage(message ? { type: 'error', text: message } : { type: '', text: '' })
          }
        />

        <div className="space-y-3">
          <PrimaryButton
            type="button"
            onClick={handleIdentify}
            disabled={!recording?.blob || isIdentifying || isSigningIn}
            className="w-full py-4 text-lg"
          >
            <Mic className="mr-2 h-4 w-4" />
            {isIdentifying ? 'Identifying...' : 'Identify Voice'}
          </PrimaryButton>

          {userGuess && (
            <div className="rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-900">
              <p className="flex items-center gap-2 font-semibold">
                <UserRound className="h-4 w-4" />
                We think you are: {userGuess.username}
              </p>
              {typeof guessScore === 'number' && (
                <p className="mt-1 text-blue-800">Similarity: {guessScore.toFixed(4)}</p>
              )}
            </div>
          )}

          {identifyResult?.temporary_login_token && (
            <form onSubmit={handleVoiceLogin} className="space-y-3 rounded-xl border border-slate-200 bg-white p-4">
              <label className="block text-sm font-medium text-slate-700" htmlFor="voice-password">
                Password
              </label>
              <input
                id="voice-password"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="Enter your password"
                className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-200"
              />

              <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-600">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(event) => setRememberMe(event.target.checked)}
                  className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-300"
                />
                Remember Me
              </label>

              <PrimaryButton
                type="submit"
                disabled={password.length < 8 || isSigningIn || isIdentifying}
                className="w-full"
              >
                {isSigningIn ? 'Signing In...' : 'Complete Voice Login'}
              </PrimaryButton>
            </form>
          )}

          <div className="min-h-12 rounded-xl border border-dashed border-slate-300 bg-slate-50 px-4 py-3 text-sm">
            {statusMessage.type === 'success' && (
              <p className="flex items-center gap-2 text-emerald-700">
                <CheckCircle2 className="h-4 w-4" />
                {statusMessage.text}
              </p>
            )}

            {statusMessage.type === 'error' && (
              <p className="flex items-center gap-2 text-rose-700">
                <AlertCircle className="h-4 w-4" />
                {statusMessage.text}
              </p>
            )}

            {!statusMessage.type && (
              <p className="text-slate-500">Record your voice to start identification.</p>
            )}
          </div>

          <p className="text-center text-sm text-slate-600">
            Prefer username and password? <Link to="/login" className="font-semibold text-blue-600">Sign in here</Link>
          </p>
        </div>
      </div>
    </main>
  )
}

export default VoiceLogin