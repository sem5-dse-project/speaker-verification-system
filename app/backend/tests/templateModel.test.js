jest.mock('../config/db', () => ({
  pool: {
    query: jest.fn(),
  },
}))

const { pool } = require('../config/db')
const templateModel = require('../models/templateModel')

describe('templateModel (pgvector)', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('upserts a 192-dim template embedding and reads it back', async () => {
    const embedding = new Array(192).fill(0).map((_, i) => (i % 2 === 0 ? 0.1 : -0.1))

    pool.query
      .mockResolvedValueOnce({ rows: [] }) // INSERT ... ON CONFLICT
      .mockResolvedValueOnce({
        rows: [
          {
            user_id: 1,
            embedding,
            embedding_dim: 192,
            num_samples: 3,
            threshold: null,
            updated_at: '2026-09-14T00:00:00.000Z',
          },
        ],
      })

    const template = await templateModel.upsertTemplate(1, embedding, 192, 3, null)

    expect(pool.query).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining('INSERT INTO speaker_embeddings'),
      expect.arrayContaining([1]),
    )
    expect(template.embedding_dim).toBe(192)
    expect(template.embedding).toHaveLength(192)
  })

  it('findBestMatch converts pgvector cosine distance into cosine similarity', async () => {
    // pgvector `<=>` returns cosine distance = 1 - cosine_similarity.
    // A returned "score" column of `1 - distance` in SQL should surface here
    // as a similarity in [-1, 1], matching the app's existing threshold math.
    pool.query.mockResolvedValueOnce({
      rows: [
        { user_id: 10, username: 'alice', score: 0.92 },
        { user_id: 11, username: 'bob', score: 0.4 },
      ],
    })

    const match = await templateModel.findBestMatch(new Array(192).fill(0.1), 2)

    expect(pool.query).toHaveBeenCalledWith(
      expect.stringContaining('embedding <=>'),
      expect.any(Array),
    )
    expect(match.user_id).toBe(10)
    expect(match.score).toBeCloseTo(0.92, 5)
    expect(match.second_score).toBeCloseTo(0.4, 5)
    expect(match.margin).toBeCloseTo(0.52, 5)
  })

  it('findBestMatch returns null when no templates are enrolled', async () => {
    pool.query.mockResolvedValueOnce({ rows: [] })

    const match = await templateModel.findBestMatch(new Array(192).fill(0.1), 2)

    expect(match).toBeNull()
  })

  it('deleteTemplateByUserId reports whether a row was removed', async () => {
    pool.query.mockResolvedValueOnce({ rowCount: 1 })
    await expect(templateModel.deleteTemplateByUserId(1)).resolves.toBe(true)

    pool.query.mockResolvedValueOnce({ rowCount: 0 })
    await expect(templateModel.deleteTemplateByUserId(2)).resolves.toBe(false)
  })
})
