import React from 'react'
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import Recorder, { reachedRecordingLimit } from './Recorder.jsx'

describe('reachedRecordingLimit', () => {
  it('stops once the clip reaches four seconds', () => {
    expect(reachedRecordingLimit(16000 * 4, 16000)).toBe(true)
    expect(reachedRecordingLimit(16000 * 4 - 1, 16000)).toBe(false)
  })
})

describe('Recorder', () => {
  it('asks for about four seconds when there is no sentence', () => {
    render(<Recorder onRecordingChange={() => {}} onRecorderError={() => {}} />)

    expect(screen.getByText(/Speak for about 4 seconds/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start Recording' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Stop Recording' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Play Recording' })).not.toBeInTheDocument()
  })

  it('mentions the sentence only when one is on screen', () => {
    render(<Recorder readsSentence onRecordingChange={() => {}} onRecorderError={() => {}} />)

    expect(screen.getByText(/Read the sentence aloud for about 4 seconds/i)).toBeInTheDocument()
  })
})
