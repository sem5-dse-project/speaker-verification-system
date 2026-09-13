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
      <label htmlFor={id} className="block text-sm font-semibold text-slate-700 dark:text-slate-300">
        {label}
      </label>

      <div
        className={[
          'group relative flex items-center rounded-xl border bg-white transition-all duration-200 dark:bg-surface-800',
          hasError
            ? 'border-rose-300 ring-2 ring-rose-100 dark:border-rose-700 dark:ring-rose-950/40'
            : 'border-brand-200 hover:border-brand-300 focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-100 dark:border-brand-900/30 dark:hover:border-brand-800/50 dark:focus-within:border-brand-600 dark:focus-within:ring-brand-950/40',
        ].join(' ')}
      >
        {Icon && (
          <Icon
            aria-hidden="true"
            className={[
              'pointer-events-none ml-3 h-4 w-4 transition-colors duration-200',
              hasError
                ? 'text-rose-400'
                : 'text-slate-400 group-focus-within:text-brand-600 dark:text-slate-500 dark:group-focus-within:text-brand-400',
            ].join(' ')}
          />
        )}

        <input
          id={id}
          type={type}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          className="w-full rounded-xl bg-transparent px-3 py-3.5 text-base text-slate-900 placeholder:text-slate-400 focus:outline-none sm:text-sm dark:text-slate-100 dark:placeholder:text-slate-500"
          aria-invalid={hasError}
          aria-describedby={helperText ? `${id}-hint` : undefined}
        />

        {rightElement && <div className="pr-3">{rightElement}</div>}
      </div>

      {helperText && (
        <p
          id={`${id}-hint`}
          className={hasError ? 'text-xs text-rose-600 dark:text-rose-400' : 'text-xs text-slate-500 dark:text-slate-400'}
        >
          {helperText}
        </p>
      )}
    </div>
  )
}

export default InputField
