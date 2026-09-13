import { useId } from 'react'

const SIZE_MAP = {
  xs: 16,
  sm: 20,
  md: 24,
  lg: 32,
  xl: 40,
}

function BrandIcon({ size = 'md', className = '' }) {
  const gradientId = useId().replace(/:/g, '')
  const px = SIZE_MAP[size] || SIZE_MAP.md

  return (
    <svg
      width={px}
      height={px}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={gradientId} x1="6" y1="4" x2="26" y2="28" gradientUnits="userSpaceOnUse">
          <stop stopColor="#863bff" />
          <stop offset="0.45" stopColor="#059669" />
          <stop offset="1" stopColor="#47bfff" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="8" fill={`url(#${gradientId})`} opacity="0.14" />
      <rect x="1" y="1" width="30" height="30" rx="8" stroke={`url(#${gradientId})`} strokeOpacity="0.35" />
      <path
        d="M16 5.5 21.5 14.5H18.2L20.8 26.5 10.5 14.5H13.8L16 5.5Z"
        fill={`url(#${gradientId})`}
      />
      <path
        d="M8.5 18.5C8.5 21.5 11.8 24 16 24C20.2 24 23.5 21.5 23.5 18.5"
        stroke={`url(#${gradientId})`}
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <path
        d="M16 24V27.5"
        stroke={`url(#${gradientId})`}
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <circle cx="16" cy="27.8" r="1.2" fill={`url(#${gradientId})`} />
    </svg>
  )
}

export default BrandIcon
