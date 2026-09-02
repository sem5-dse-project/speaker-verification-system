import { useCallback, useEffect, useMemo, useState } from 'react'
import { Download, LogOut, Mic, ShieldPlus, UploadCloud } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import Recorder from '../components/Recorder.jsx'
import PrimaryButton from '../components/PrimaryButton.jsx'
import PageShell from '../components/PageShell.jsx'
import SentenceCard from '../components/SentenceCard.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import api from '../services/api.js'
import {
  COLLECTION_PHRASES,
  pickCollectionPhrase,
  phrasesFromSamples,
} from '../utils/collectionPhrases.js'

const INITIAL_FORM = {
  speaker_id: '',
  label: 'live',
  phrase: '',
  phone_model: '',
  distance: '',
  volume: '',
  notes: '',
  consent: false,
}

const INITIAL_ADMIN_FORM = {
  username: '',
  password: '',
}

function AdminDashboard() {
  const navigate = useNavigate()
  const { user, logout } = useAuth()
  const [recording, setRecording] = useState(null)
  const [form, setForm] = useState(INITIAL_FORM)
  const [adminForm, setAdminForm] = useState(INITIAL_ADMIN_FORM)
  const [samples, setSamples] = useState([])
  const [counts, setCounts] = useState([])
  const [admins, setAdmins] = useState([])
  const [recorderKey, setRecorderKey] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [isCreatingAdmin, setIsCreatingAdmin] = useState(false)
  const [statusMessage, setStatusMessage] = useState({ type: '', text: '' })

  const usedPhrases = useMemo(() => phrasesFromSamples(samples), [samples])
  const phrasesRemaining = COLLECTION_PHRASES.length - usedPhrases.size

  const loadDashboard = useCallback(async () => {
    try {
      const [collectionRes, adminsRes] = await Promise.all([
        api.get('/admin/collection'),
        api.get('/admin/admins'),
      ])
      setSamples(collectionRes.data.samples || [])
      setCounts(collectionRes.data.counts || [])
      setAdmins(adminsRes.data.admins || [])
    } catch (error) {
      setStatusMessage({
        type: 'error',
        text: error.response?.data?.message || 'Failed to load admin dashboard.',
      })
    }
  }, [])

  useEffect(() => {
    loadDashboard()
  }, [loadDashboard])

  useEffect(() => {
    if (form.phrase) {
      return
    }

    const { phrase, exhausted } = pickCollectionPhrase(usedPhrases)
    if (exhausted) {
      setStatusMessage({
        type: 'error',
        text: 'All collection phrases have been used. Export data before collecting more.',
      })
      return
    }

    setForm((previous) => ({ ...previous, phrase }))
  }, [form.phrase, usedPhrases])

  const handleGeneratePhrase = () => {
    const { phrase, exhausted } = pickCollectionPhrase(usedPhrases, form.phrase)
    if (exhausted) {
      setStatusMessage({
        type: 'error',
        text: 'No unused phrases left. Export or archive samples before generating another.',
      })
      return
    }

    setForm((previous) => ({ ...previous, phrase }))
    setRecording(null)
    setRecorderKey((key) => key + 1)
    setStatusMessage({ type: '', text: '' })
  }

  const handleLogout = () => {
    logout()
    navigate('/admin/login', { replace: true })
  }

  const handleFormChange = (field) => (event) => {
    const value =
      event.target.type === 'checkbox' ? event.target.checked : event.target.value
    setForm((previous) => ({ ...previous, [field]: value }))
  }

  const handleAdminFormChange = (field) => (event) => {
    setAdminForm((previous) => ({ ...previous, [field]: event.target.value }))
  }

  const handleUploadSample = async () => {
    if (!recording?.blob) {
      setStatusMessage({ type: 'error', text: 'Record audio before uploading.' })
      return
    }

    if (!form.speaker_id.trim()) {
      setStatusMessage({ type: 'error', text: 'Speaker ID is required.' })
      return
    }

    if (!form.consent) {
      setStatusMessage({ type: 'error', text: 'Consent is required for research collection.' })
      return
    }

    if (!form.phrase.trim()) {
      setStatusMessage({
        type: 'error',
        text: 'No phrase assigned. Generate a new sentence before recording.',
      })
      return
    }

    setIsUploading(true)
    setStatusMessage({ type: '', text: '' })

    try {
      const payload = new FormData()
      const wavFile = new File([recording.blob], `collection_${Date.now()}.wav`, {
        type: 'audio/wav',
      })
      payload.append('audio', wavFile)
      payload.append('speaker_id', form.speaker_id.trim())
      payload.append('label', form.label)
      payload.append('phrase', form.phrase)
      payload.append('phone_model', form.phone_model)
      payload.append('distance', form.distance)
      payload.append('volume', form.volume)
      payload.append('notes', form.notes)
      payload.append('consent', 'true')

      const response = await api.post('/admin/collection', payload)
      setCounts(response.data.counts || [])
      setStatusMessage({
        type: 'success',
        text: `Saved ${response.data.sample.label} sample for ${response.data.sample.speaker_id}. Replay score: ${
          response.data.sample.replay_score ?? 'n/a'
        }`,
      })
      setRecording(null)
      setRecorderKey((key) => key + 1)
      setForm((previous) => ({
        ...INITIAL_FORM,
        speaker_id: previous.speaker_id,
        label: previous.label,
      }))
      await loadDashboard()
    } catch (error) {
      setStatusMessage({
        type: 'error',
        text: error.response?.data?.message || 'Failed to upload collection sample.',
      })
    } finally {
      setIsUploading(false)
    }
  }

  const handleCreateAdmin = async (event) => {
    event.preventDefault()
    setIsCreatingAdmin(true)
    setStatusMessage({ type: '', text: '' })

    try {
      await api.post('/admin/admins', adminForm)
      setAdminForm(INITIAL_ADMIN_FORM)
      setStatusMessage({ type: 'success', text: 'New admin account created.' })
      await loadDashboard()
    } catch (error) {
      setStatusMessage({
        type: 'error',
        text: error.response?.data?.message || 'Failed to create admin.',
      })
    } finally {
      setIsCreatingAdmin(false)
    }
  }

  const handleExportCsv = async () => {
    try {
      const response = await api.get('/admin/collection/export', {
        responseType: 'blob',
      })
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', 'collection_metadata.csv')
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      setStatusMessage({
        type: 'error',
        text: error.response?.data?.message || 'Failed to export metadata.',
      })
    }
  }

  const liveCount = counts.find((row) => row.label === 'live')?.count || 0
  const replayCount = counts.find((row) => row.label === 'replay')?.count || 0

  return (
    <PageShell variant="admin">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="card flex flex-wrap items-center justify-between gap-3 p-6">
          <div>
            <h1 className="heading-1 text-3xl">Admin Data Collection</h1>
            <p className="mt-1 text-muted">
              Signed in as {user?.username}. Collect live vs phone-replay WAVs for model fine-tuning.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Link to="/dashboard" className="btn-secondary px-4 py-2 font-semibold">
              User Dashboard
            </Link>
            <button
              type="button"
              onClick={handleLogout}
              className="inline-flex items-center gap-2 rounded-xl bg-rose-600 px-4 py-2 text-sm font-semibold text-white hover:bg-rose-500 dark:bg-rose-700 dark:hover:bg-rose-600"
            >
              <LogOut className="h-4 w-4" />
              Logout
            </button>
          </div>
        </header>

        <section className="grid gap-4 sm:grid-cols-3">
          <div className="card p-5">
            <p className="text-sm text-subtle">Live samples</p>
            <p className="text-3xl font-bold text-emerald-600 dark:text-emerald-400">{liveCount}</p>
          </div>
          <div className="card p-5">
            <p className="text-sm text-subtle">Replay samples</p>
            <p className="text-3xl font-bold text-amber-600 dark:text-amber-400">{replayCount}</p>
          </div>
          <div className="card p-5">
            <p className="text-sm text-subtle">Admins</p>
            <p className="text-3xl font-bold text-brand-600 dark:text-brand-400">{admins.length}</p>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <div className="card space-y-4 p-6">
            <h2 className="heading-2">Collect Sample</h2>

            <p className="text-sm text-muted">
              {phrasesRemaining} of {COLLECTION_PHRASES.length} phrases still available (longer than
              enrollment sentences; each phrase is used once).
            </p>

            {form.phrase ? (
              <SentenceCard sentence={form.phrase} onGenerateSentence={handleGeneratePhrase} />
            ) : (
              <p className="alert-warning">
                All phrases have been collected. Export CSV and archive samples to start a new batch.
              </p>
            )}

            <div className="grid gap-3 sm:grid-cols-2">
              <label className="block text-sm">
                <span className="label-text">Speaker ID</span>
                <input
                  value={form.speaker_id}
                  onChange={handleFormChange('speaker_id')}
                  placeholder="spk01"
                  className="input-field"
                />
              </label>

              <label className="block text-sm">
                <span className="label-text">Label</span>
                <select
                  value={form.label}
                  onChange={handleFormChange('label')}
                  className="input-field"
                >
                  <option value="live">Live mic</option>
                  <option value="replay">Phone replay</option>
                </select>
              </label>

              <label className="block text-sm">
                <span className="label-text">Phone model</span>
                <input
                  value={form.phone_model}
                  onChange={handleFormChange('phone_model')}
                  placeholder="Samsung A54"
                  className="input-field"
                />
              </label>

              <label className="block text-sm">
                <span className="label-text">Distance / volume</span>
                <div className="grid grid-cols-2 gap-2">
                  <input
                    value={form.distance}
                    onChange={handleFormChange('distance')}
                    placeholder="30 cm"
                    className="input-field"
                  />
                  <input
                    value={form.volume}
                    onChange={handleFormChange('volume')}
                    placeholder="medium"
                    className="input-field"
                  />
                </div>
              </label>

              <label className="block text-sm sm:col-span-2">
                <span className="label-text">Notes</span>
                <textarea
                  value={form.notes}
                  onChange={handleFormChange('notes')}
                  rows={2}
                  placeholder="Room noise, playback source, etc."
                  className="input-field"
                />
              </label>
            </div>

            <label className="flex items-start gap-2 text-sm text-body">
              <input
                type="checkbox"
                checked={form.consent}
                onChange={handleFormChange('consent')}
                className="mt-1 checkbox-brand"
              />
              Speaker consented to use this recording for research and model improvement.
            </label>

            <Recorder
              key={recorderKey}
              onRecordingChange={setRecording}
              onRecorderError={(message) =>
                setStatusMessage(message ? { type: 'error', text: message } : { type: '', text: '' })
              }
            />

            <PrimaryButton type="button" onClick={handleUploadSample} disabled={isUploading}>
              <UploadCloud className="mr-2 h-4 w-4" />
              {isUploading ? 'Saving Sample...' : 'Save Collection Sample'}
            </PrimaryButton>
          </div>

          <div className="space-y-6">
            <div className="card p-6">
              <div className="mb-4 flex items-center justify-between gap-3">
                <h2 className="heading-2">Recent Samples</h2>
                <button type="button" onClick={handleExportCsv} className="btn-secondary font-semibold">
                  <Download className="h-4 w-4" />
                  Export CSV
                </button>
              </div>

              <div className="max-h-80 space-y-2 overflow-y-auto">
                {samples.length === 0 && (
                  <p className="text-sm text-subtle">No collection samples yet.</p>
                )}
                {samples.slice(0, 20).map((sample) => (
                  <div key={sample.id} className="list-item">
                    <p className="font-semibold">
                      {sample.speaker_id} · {sample.label.toUpperCase()}
                    </p>
                    <p className="text-xs text-subtle">
                      score {sample.replay_score ?? 'n/a'} · {sample.replay_decision ?? 'n/a'} ·{' '}
                      {sample.created_at}
                    </p>
                    {sample.phrase && (
                      <p className="mt-1 line-clamp-2 text-xs text-muted">{sample.phrase}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="card p-6">
              <h2 className="mb-4 flex items-center gap-2 heading-2">
                <ShieldPlus className="h-5 w-5" />
                Create Admin
              </h2>

              <form className="space-y-3" onSubmit={handleCreateAdmin}>
                <input
                  value={adminForm.username}
                  onChange={handleAdminFormChange('username')}
                  placeholder="New admin username"
                  className="input-field"
                />
                <input
                  type="password"
                  value={adminForm.password}
                  onChange={handleAdminFormChange('password')}
                  placeholder="Password (min 6 chars)"
                  className="input-field"
                />
                <PrimaryButton type="submit" disabled={isCreatingAdmin}>
                  {isCreatingAdmin ? 'Creating...' : 'Create Admin Account'}
                </PrimaryButton>
              </form>

              <div className="mt-4 space-y-1 text-sm text-muted">
                {admins.map((admin) => (
                  <p key={admin.id}>
                    {admin.username} · joined {admin.created_at}
                  </p>
                ))}
              </div>
            </div>
          </div>
        </section>

        {statusMessage.text && (
          <p className={statusMessage.type === 'success' ? 'alert-success' : 'alert-error'}>
            {statusMessage.text}
          </p>
        )}

        <p className="flex items-center gap-2 text-sm text-subtle">
          <Mic className="h-4 w-4" />
          WAV files are stored locally under uploads/collection and are gitignored.
        </p>
      </div>
    </PageShell>
  )
}

export default AdminDashboard
