/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'node',
  roots: ['<rootDir>/tests/integration'],
  testMatch: ['**/*.test.js'],
  verbose: true,
  // These tests hit a real PostgreSQL instance (docker-compose) rather than
  // mocking config/db, so run them serially and allow more time per test.
  maxWorkers: 1,
  testTimeout: 15000,
}
