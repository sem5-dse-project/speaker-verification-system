const mockRes = () => {
  const res = {}
  res.status = jest.fn().mockReturnValue(res)
  res.json = jest.fn().mockReturnValue(res)
  return res
}

const mockNext = () => jest.fn()

module.exports = {
  mockRes,
  mockNext,
}
