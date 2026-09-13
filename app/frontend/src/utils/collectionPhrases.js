/** Longer read-aloud phrases for admin live/replay data collection. */

export const COLLECTION_PHRASES = [
  'Before we begin the voice recording session, please confirm that you understand this sample will be stored locally for replay detection research only.',
  'When collecting live microphone audio, speak at a natural pace in a quiet room and avoid covering the laptop microphone with your hand.',
  'For phone replay samples, play the enrolled phrase from the mobile speaker while the laptop records from a fixed distance without talking at the same time.',
  'Reliable anti-spoof systems must distinguish between genuine speech and playback attacks captured through different microphones, rooms, and recording devices.',
  'The inverted Mel replay detector learns patterns from spectrograms, so consistent pronunciation and minimal background noise improve the quality of collected data.',
  'Researchers use balanced datasets of live and replay utterances to fine-tune thresholds that reduce false rejections while still blocking impersonation attempts.',
  'Each speaker should use a unique identifier, read the assigned sentence clearly, and wait for the upload confirmation before starting the next recording.',
  'Voice authentication combines speaker verification with replay detection, meaning the system first checks whether the audio is live before matching the voice template.',
  'During enrollment users record short sentences, but collection sessions require longer phrases so replay and live samples contain more acoustic detail for training.',
  'If the room becomes noisy during recording, pause and try again rather than submitting a clip where traffic, fans, or conversations dominate the waveform.',
  'Physical replay attacks often introduce subtle high-frequency artifacts from phone speakers, which is why inverted Mel features emphasize those bands during classification.',
  'After saving a live sample, repeat the same phrase from a phone speaker for the replay label while keeping the laptop microphone position unchanged between takes.',
  'Consent is required for every collected clip because personal voice recordings must remain on local storage and must never be committed to public repositories.',
  'The admin panel automatically assigns unused sentences so collectors do not repeat the same phrase until the full pool has been recorded at least once.',
  'Clear articulation, stable volume, and at least three seconds of continuous speech help both Silero voice activity detection and the downstream replay scoring pipeline.',
  'Synthetic spoof detectors trained on broadcast datasets may fail on browser microphones, so local collection closes the gap between laboratory benchmarks and real application audio.',
  'Document the phone model, playback volume, and distance in the notes field whenever you capture replay samples so experiments remain reproducible during later analysis.',
  'When exporting metadata to CSV, each row links the WAV file path, speaker code, label, assigned phrase, and optional replay score from the deployed detection model.',
]

export const pickCollectionPhrase = (usedPhrases, exclude = null) => {
  const used = usedPhrases instanceof Set ? usedPhrases : new Set(usedPhrases)
  if (exclude) {
    used.add(exclude)
  }

  const available = COLLECTION_PHRASES.filter((phrase) => !used.has(phrase))
  if (available.length === 0) {
    return { phrase: null, exhausted: true }
  }

  const phrase = available[Math.floor(Math.random() * available.length)]
  return { phrase, exhausted: false }
}

export const phrasesFromSamples = (samples) =>
  new Set(
    (samples || [])
      .map((sample) => sample.phrase)
      .filter((phrase) => phrase && COLLECTION_PHRASES.includes(phrase)),
  )
