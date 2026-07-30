import { describe, it, expect } from 'vitest'
import { getStrengthFromPassword } from '../components/PasswordStrength.jsx'

describe('getStrengthFromPassword', () => {
  it('marks short passwords as Weak', () => {
    expect(getStrengthFromPassword('abc')).toEqual(
      expect.objectContaining({ label: 'Weak', level: 1 }),
    )
  })

  it('marks mixed medium passwords as Medium', () => {
    expect(getStrengthFromPassword('Abcdefgh')).toEqual(
      expect.objectContaining({ label: 'Medium', level: 2 }),
    )
  })

  it('marks strong passwords as Strong', () => {
    expect(getStrengthFromPassword('Abcdefg1')).toEqual(
      expect.objectContaining({ label: 'Strong', level: 3 }),
    )
  })
})
