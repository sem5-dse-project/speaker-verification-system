import React from 'react'

function PrimaryButton({ children, className = '', ...props }) {
  return (
    <button
      className={`inline-flex items-center justify-center rounded-2xl bg-blue-600 px-6 py-3 text-base font-semibold text-white shadow-lg shadow-blue-200 transition-all duration-200 hover:-translate-y-0.5 hover:bg-blue-500 hover:shadow-xl hover:shadow-blue-200/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:translate-y-0 disabled:bg-slate-300 disabled:text-slate-500 disabled:shadow-none dark:shadow-blue-950/30 dark:hover:shadow-blue-950/40 dark:disabled:bg-slate-700 dark:disabled:text-slate-500 ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}

export default PrimaryButton
