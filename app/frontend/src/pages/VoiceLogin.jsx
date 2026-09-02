import { useState } from 'react'
import { AlertCircle, CheckCircle2, Mic, UserRound } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import PageShell from '../components/PageShell.jsx'
import PageHeader from '../components/PageHeader.jsx'
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
    <PageShell narrow>
      <div className="space-y-5 sm:space-y-6">
        <PageHeader
          icon={Mic}
          title="Login with Voice"
          subtitle="Record once, pass replay detection, then confirm with your password."
          action={
            <Link to="/login" className="btn-secondary">
              Password login
            </Link>
          }
        />

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
            <div className="alert-info">
              <p className="flex items-center gap-2 font-semibold">
                <UserRound className="h-4 w-4" />
                We think you are: {userGuess.username}
              </p>
              {typeof guessScore === 'number' && (
                <p className="mt-1 opacity-90">Similarity: {guessScore.toFixed(4)}</p>
              )}
            </div>
          )}

          {identifyResult?.temporary_login_token && (
            <form onSubmit={handleVoiceLogin} className="card space-y-3 p-4">
              <label className="label-text text-sm" htmlFor="voice-password">
                Password
              </label>
              <input
                id="voice-password"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="Enter your password"
                className="input-field"
              />

              <label className="flex cursor-pointer items-center gap-2 text-sm text-muted">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(event) => setRememberMe(event.target.checked)}
                  className="checkbox-brand"
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

          <div className="status-panel">
            {statusMessage.type === 'success' && (
              <p className="flex items-center gap-2 text-emerald-700 dark:text-emerald-300">
                <CheckCircle2 className="h-4 w-4" />
                {statusMessage.text}
              </p>
            )}

            {statusMessage.type === 'error' && (
              <p className="flex items-center gap-2 text-rose-700 dark:text-rose-300">
                <AlertCircle className="h-4 w-4" />
                {statusMessage.text}
              </p>
            )}

            {!statusMessage.type && (
              <p className="text-subtle">Record your voice to start identification.</p>
            )}
          </div>

          <p className="text-center text-sm text-muted">
            Prefer username and password?{' '}
            <Link to="/login" className="link-primary">
              Sign in here
            </Link>
          </p>
        </div>
      </div>
    </PageShell>
  )
}

export default VoiceLogin
