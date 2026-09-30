/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'node',
  roots: ['<rootDir>/tests'],
  testMatch: ['**/*.test.js'],
  // Integration suite hits a real database and has its own config/run script
  // (npm run test:integration) - keep it out of the default mocked unit run.
  testPathIgnorePatterns: ['<rootDir>/tests/integration/'],
  clearMocks: true,
  verbose: true,
}
