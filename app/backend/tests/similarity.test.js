const {
  cosineSimilarity,
  findBestTemplateMatch,
  passesIdentifyGate,
} = require('../services/similarity')

describe('similarity identify helpers', () => {
  it('ranks best and second-best template scores', () => {
    const match = findBestTemplateMatch([1, 0], [
      { user_id: 1, username: 'a', embedding: [1, 0] },
      { user_id: 2, username: 'b', embedding: [0.8, 0.6] },
      { user_id: 3, username: 'c', embedding: [0, 1] },
    ])

    expect(match.user_id).toBe(1)
    expect(match.score).toBeCloseTo(1, 5)
    expect(match.second_score).toBeGreaterThan(0.5)
    expect(match.margin).toBeCloseTo(match.score - match.second_score, 5)
  })

  it('rejects scores below threshold', () => {
    const gate = passesIdentifyGate(
      { score: 0.1, second_score: null, margin: null },
      { threshold: 0.25, margin: 0.05 },
    )
    expect(gate.accepted).toBe(false)
    expect(gate.reason).toBe('below_threshold')
  })

  it('rejects ambiguous top-two matches', () => {
    const gate = passesIdentifyGate(
      { score: 0.9, second_score: 0.88, margin: 0.02 },
      { threshold: 0.25, margin: 0.05 },
    )
    expect(gate.accepted).toBe(false)
    expect(gate.reason).toBe('ambiguous')
  })

  it('accepts a clear top match', () => {
    const gate = passesIdentifyGate(
      { score: 0.9, second_score: 0.5, margin: 0.4 },
      { threshold: 0.25, margin: 0.05 },
    )
    expect(gate.accepted).toBe(true)
    expect(cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1, 5)
  })
})
