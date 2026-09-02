import BrandIcon from './BrandIcon.jsx'

function BrandMark({
  size = 'md',
  subtitle,
  variant = 'default',
  className = '',
  showIcon = true,
}) {
  const titleClass =
    variant === 'light'
      ? 'brand-mark-title brand-mark-title-light'
      : variant === 'on-dark'
        ? 'brand-mark-title brand-mark-title-on-dark'
        : 'brand-mark-title'

  const subtitleClass =
    variant === 'light' || variant === 'on-dark'
      ? 'brand-mark-subtitle brand-mark-subtitle-light'
      : 'brand-mark-subtitle'

  return (
    <div className={`brand-mark ${className}`.trim()}>
      {showIcon && <BrandIcon size={size} className="shrink-0" />}
      <div className="min-w-0">
        <p className={titleClass}>Voice Authentication</p>
        {subtitle && <p className={subtitleClass}>{subtitle}</p>}
      </div>
    </div>
  )
}

export default BrandMark
