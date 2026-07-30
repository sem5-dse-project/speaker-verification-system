import React from 'react'
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import PrimaryButton from './PrimaryButton.jsx'

describe('PrimaryButton', () => {
  it('renders children and handles click', () => {
    const onClick = vi.fn()
    render(<PrimaryButton onClick={onClick}>Verify Voice</PrimaryButton>)

    fireEvent.click(screen.getByRole('button', { name: 'Verify Voice' }))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('respects disabled state', () => {
    const onClick = vi.fn()
    render(
      <PrimaryButton disabled onClick={onClick}>
        Submit
      </PrimaryButton>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Submit' }))
    expect(onClick).not.toHaveBeenCalled()
  })
})
