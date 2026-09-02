import React from 'react'

function PrimaryButton({ children, className = '', ...props }) {
  return (
    <button className={`btn-primary ${className}`} {...props}>
      {children}
    </button>
  )
}

export default PrimaryButton
