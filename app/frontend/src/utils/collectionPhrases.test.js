import { describe, expect, it } from 'vitest'
import {
  COLLECTION_PHRASES,
  pickCollectionPhrase,
  phrasesFromSamples,
} from './collectionPhrases.js'

describe('collectionPhrases', () => {
  it('uses longer phrases than typical enrollment sentences', () => {
    const minWords = Math.min(
      ...COLLECTION_PHRASES.map((phrase) => phrase.split(/\s+/).length),
    )
    expect(minWords).toBeGreaterThan(12)
  })

  it('does not repeat a phrase already marked as used', () => {
    const used = new Set([COLLECTION_PHRASES[0], COLLECTION_PHRASES[1]])
    const { phrase, exhausted } = pickCollectionPhrase(used)
    expect(exhausted).toBe(false)
    expect(used.has(phrase)).toBe(false)
  })

  it('reports exhaustion when every phrase was used', () => {
    const used = new Set(COLLECTION_PHRASES)
    const { phrase, exhausted } = pickCollectionPhrase(used)
    expect(phrase).toBeNull()
    expect(exhausted).toBe(true)
  })

  it('collects known phrases from saved samples', () => {
    const used = phrasesFromSamples([
      { phrase: COLLECTION_PHRASES[0] },
      { phrase: 'custom manual note' },
    ])
    expect(used.size).toBe(1)
    expect(used.has(COLLECTION_PHRASES[0])).toBe(true)
  })
})
