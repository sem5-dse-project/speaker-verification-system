function InputField({
  id,
  label,
  type = 'text',
  value,
  onChange,
  placeholder,
  icon: Icon,
  error,
  helperText,
  rightElement,
}) {
  const hasError = Boolean(error)

  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="block text-sm font-semibold text-slate-700">
        {label}
      </label>

      <div
        className={[
          'group relative flex items-center rounded-xl border bg-white transition-all duration-200',
          hasError
            ? 'border-rose-300 ring-2 ring-rose-100'
            : 'border-slate-200 hover:border-slate-300 focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100',
        ].join(' ')}
      >
        {Icon && (
          <Icon
            aria-hidden="true"
            className={[
              'pointer-events-none ml-3 h-4 w-4 transition-colors duration-200',
              hasError ? 'text-rose-400' : 'text-slate-400 group-focus-within:text-blue-500',
            ].join(' ')}
          />
        )}

        <input
          id={id}
          type={type}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          className="w-full rounded-xl bg-transparent px-3 py-3.5 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none"
          aria-invalid={hasError}
          aria-describedby={helperText ? `${id}-hint` : undefined}
        />

        {rightElement && <div className="pr-3">{rightElement}</div>}
      </div>

      {helperText && (
        <p
          id={`${id}-hint`}
          className={hasError ? 'text-xs text-rose-600' : 'text-xs text-slate-500'}
        >
          {helperText}
        </p>
      )}
    </div>
  )
}

export default InputField