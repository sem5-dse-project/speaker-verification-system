import '@testing-library/jest-dom/vitest'

// jsdom Blob lacks arrayBuffer(); FileReader works there.
if (typeof Blob !== 'undefined' && typeof Blob.prototype.arrayBuffer !== 'function') {
  Blob.prototype.arrayBuffer = function arrayBuffer() {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result)
      reader.onerror = () => reject(reader.error || new Error('Failed to read blob'))
      reader.readAsArrayBuffer(this)
    })
  }
}
