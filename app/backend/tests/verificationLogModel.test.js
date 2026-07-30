jest.mock('../config/db', () => ({
  pool: {
    query: jest.fn(),
  },
}))

const { pool } = require('../config/db')
const {
  createVerificationLog,
  getVerificationLogsByUserId,
} = require('../models/verificationLogModel')

describe('verificationLogModel', () => {
  it('inserts a log and returns the normalized row', async () => {
    pool.query
      .mockResolvedValueOnce([{ insertId: 9 }])
      .mockResolvedValueOnce([
        [
          {
            id: 9,
            user_id: 1,
            voice_sample_id: 4,
            score: '0.71',
            threshold: '0.25',
            accepted: 1,
            decision: 'ACCEPT',
            created_at: '2026-07-30T00:00:00.000Z',
          },
        ],
      ])

    const log = await createVerificationLog({
      userId: 1,
      voiceSampleId: 4,
      score: 0.71,
      threshold: 0.25,
      accepted: true,
      decision: 'ACCEPT',
    })

    expect(pool.query).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining('INSERT INTO verification_logs'),
      [1, 4, 0.71, 0.25, 1, 'ACCEPT'],
    )
    expect(log).toEqual(
      expect.objectContaining({
        id: 9,
        accepted: true,
        score: 0.71,
        decision: 'ACCEPT',
      }),
    )
  })

  it('lists logs for a user with numeric coercion', async () => {
    pool.query.mockResolvedValueOnce([
      [
        {
          id: 1,
          user_id: 2,
          voice_sample_id: 3,
          score: '0.1',
          threshold: '0.25',
          accepted: 0,
          decision: 'REJECT',
          created_at: '2026-07-30T00:00:00.000Z',
          file_path: 'uploads/verifications/user_2/a.wav',
        },
      ],
    ])

    const logs = await getVerificationLogsByUserId(2, 10)

    expect(pool.query).toHaveBeenCalledWith(
      expect.stringContaining('FROM verification_logs'),
      [2, 10],
    )
    expect(logs[0]).toEqual(
      expect.objectContaining({
        accepted: false,
        score: 0.1,
        file_path: 'uploads/verifications/user_2/a.wav',
      }),
    )
  })
})
