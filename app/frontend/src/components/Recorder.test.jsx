import React from 'react'
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import Recorder from './Recorder.jsx'

describe('Recorder', () => {
  it('asks the user to stop the recording when there is no sentence', () => {
    render(<Recorder onRecordingChange={() => {}} onRecorderError={() => {}} />)

    expect(screen.getByText(/then stop the recording/i)).toBeInTheDocument()
    expect(screen.queryByText(/4 seconds/i)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start Recording' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Stop Recording' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Play Recording' })).not.toBeInTheDocument()
  })

  it('mentions the sentence only when one is on screen', () => {
    render(<Recorder readsSentence onRecordingChange={() => {}} onRecorderError={() => {}} />)

    expect(screen.getByText(/Read the sentence aloud/i)).toBeInTheDocument()
  })
})
