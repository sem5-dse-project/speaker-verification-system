import React from 'react'
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import StatusBadge from './StatusBadge.jsx'

describe('StatusBadge', () => {
  it('renders Ready by default / unknown status', () => {
    render(<StatusBadge status="unknown" />)
    expect(screen.getByText('Ready')).toBeInTheDocument()
  })

  it('renders Recording status', () => {
    render(<StatusBadge status="recording" />)
    expect(screen.getByText('Recording...')).toBeInTheDocument()
  })

  it('renders complete status', () => {
    render(<StatusBadge status="complete" />)
    expect(screen.getByText('Recording Complete')).toBeInTheDocument()
  })
})
