import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import EnrolledRoute from './EnrolledRoute.jsx'
import api from '../services/api.js'

vi.mock('../services/api.js', () => ({
  default: {
    get: vi.fn(),
  },
}))

function renderAtVerification() {
  return render(
    <MemoryRouter initialEntries={['/verification']}>
      <Routes>
        <Route
          path="/verification"
          element={
            <EnrolledRoute>
              <p>Verify page</p>
            </EnrolledRoute>
          }
        />
        <Route path="/enrollment" element={<p>Enrollment page</p>} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('EnrolledRoute', () => {
  beforeEach(() => {
    api.get.mockReset()
  })

  it('sends an unfinished enrollment to the enrollment page', async () => {
    api.get.mockResolvedValue({ data: { template: null } })
    renderAtVerification()

    expect(await screen.findByText('Enrollment page')).toBeInTheDocument()
    expect(screen.queryByText('Verify page')).not.toBeInTheDocument()
  })

  it('shows verification when a voice template exists', async () => {
    api.get.mockResolvedValue({ data: { template: { has_embedding: true } } })
    renderAtVerification()

    expect(await screen.findByText('Verify page')).toBeInTheDocument()
  })
})
