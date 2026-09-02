const BAR_COUNT = 40

function AudioWaveform({ levels, active }) {
  const bars = levels?.length === BAR_COUNT ? levels : Array(BAR_COUNT).fill(0)

  return (
    <div
      className={`audio-waveform ${active ? 'audio-waveform-active' : ''}`}
      aria-hidden="true"
    >
      {bars.map((level, index) => (
        <span
          key={`bar-${index}`}
          className="audio-waveform-bar"
          style={{ transform: `scaleY(${Math.max(0.08, level)})` }}
        />
      ))}
    </div>
  )
}

export { BAR_COUNT }
export default AudioWaveform
